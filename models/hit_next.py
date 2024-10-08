import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from .utils import (
    does_conv2d_change_dim,
    to_tuple,
    compute_conv2d_output_resolution_from_module,
)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def img_to_seq(img):
    r"""
    Args:
        img (tensor): (B, H, W, C)

    Returns:
        seq (tensor): (B, H*W, C)
    """

    B, H, W, C = img.shape

    return img.view(B, H * W, C)


def seq_to_img(seq, img_resolution):
    r"""
    Args:
        seq (tensor): (B, H*W, C)
        img_resolution (tuple): (H, W)

    Returns:
        img (tensor): (B, H, W, C)
    """

    B, L, C = seq.shape
    H, W = img_resolution
    assert L == H * W, "input seq has wrong size"

    return seq.view(B, H, W, C)


def window_partition(
    x,
    window_size,
):
    r"""
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows,
    window_size,
    H,
    W,
):
    r"""
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    r"""GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class CoPE1d(nn.Module):
    r"""
    Args:
        npos_max (int): max size of position emb (> N = max pos value)
        head_dim (int): transformer head dimension

    Returns:
        logits: (B, nH, N, N)
    """

    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        # compute positions
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)

        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor

        return logits_ceil * w + logits_floor * (1 - w)


class CoPE2d(nn.Module):
    r"""
    Args:
        npos_max (int): max size of position emb (> (N + Ww) = (Wh*Ww + Ww) = max pos value)
        head_dim (int): transformer head dimension
        window_size (tuple[int]): The height and width of the window.

    Returns:
        logits: (B, nH, N, N)
    """

    def __init__(self, npos_max, head_dim, window_size):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))
        self.window_size = window_size

    def forward(self, query, attn_logits):
        B, nH, N, C = query.shape

        # compute positions
        gates = torch.sigmoid(attn_logits)

        # reshape to respect window dimensions
        gates_h = gates.transpose(-1, -2).reshape(
            B,
            nH,
            N,
            self.window_size[1],
            self.window_size[0],
        )  # B, nH, N, Ww, Wh

        gates_w = gates.reshape(
            B,
            nH,
            N,
            self.window_size[0],
            self.window_size[1],
        )  # B, nH, N, Wh, Ww

        # compute positions
        pos_h = gates_h.flip(-1).cumsum(dim=-1).flip(-1)
        pos_w = gates_w.flip(-1).cumsum(dim=-1).flip(-1)

        # revert to original shape
        pos_h = pos_h.reshape(
            B,
            nH,
            N,
            N,
        ).transpose(-1, -2)

        pos_w = pos_w.reshape(
            B,
            nH,
            N,
            N,
        )

        # sum positions (ensure unique sums)
        pos_h *= self.window_size[1]
        pos = pos_h + pos_w
        pos = pos.clamp(max=self.npos_max - 1)

        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor

        return logits_ceil * w + logits_floor * (1 - w)


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int | tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (int | tuple[int]): The height and width of the window in pre-training.
        rpe_type (tuple[str]): combination of relative position enconding to use
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        rpe_type=[
            "rpe_default",
            "cope_1d",
        ],  # options are rpe_default, cope_1d and cope_2d
    ):

        super().__init__()
        self.dim = dim
        self.window_size = to_tuple(window_size, 2)  # Wh, Ww
        self.pretrained_window_size = to_tuple(pretrained_window_size, 2)
        self.num_heads = num_heads
        self.rpe_type = rpe_type

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        if "rpe_default" in self.rpe_type:
            # get relative_coords_table
            relative_coords_h = torch.arange(
                -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
            )
            relative_coords_w = torch.arange(
                -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
            )
            relative_coords_table = (
                torch.stack(
                    torch.meshgrid(
                        [relative_coords_h, relative_coords_w], indexing="ij"
                    )
                )
                .permute(1, 2, 0)
                .contiguous()
                .unsqueeze(0)
            )  # 1, 2*Wh-1, 2*Ww-1, 2
            if self.pretrained_window_size[0] > 0:
                relative_coords_table[:, :, :, 0] /= self.pretrained_window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= self.pretrained_window_size[1] - 1
            else:
                relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
                relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = (
                torch.sign(relative_coords_table)
                * torch.log2(torch.abs(relative_coords_table) + 1.0)
                / np.log2(8)
            )
            self.register_buffer("relative_coords_table", relative_coords_table)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(
                torch.meshgrid([coords_h, coords_w], indexing="ij")
            )  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        if "cope_1d" in self.rpe_type:
            self.cope_1d = CoPE1d(
                npos_max=self.window_size[0] + self.window_size[1] + 1,
                head_dim=self.dim // self.num_heads,
            )

        if "cope_2d" in self.rpe_type:
            self.cope_2d = CoPE2d(
                npos_max=self.window_size[0] + self.window_size[1] + 1,
                head_dim=self.dim // self.num_heads,
                window_size=self.window_size,
            )

        if len(self.rpe_type) > 1:
            self.rpe_comb = nn.Linear(len(self.rpe_type), 1, bias=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def get_rpe(
        self,
        batch_size,
        query,
        attn_logits,
    ):
        if len(self.rpe_type) == 0:
            return 0

        rpe_list = []

        if "rpe_default" in self.rpe_type:
            relative_position_bias_table = self.cpb_mlp(
                self.relative_coords_table
            ).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            rpe_list.append(
                relative_position_bias.repeat(batch_size, 1, 1, 1)
            )  # B, nH, Wh*Ww, Wh*Ww

        if "cope_1d" in self.rpe_type:
            cope_1d = self.cope_1d(query, attn_logits)
            rpe_list.append(cope_1d)  # B, nH, Wh*Ww, Wh*Ww

        if "cope_2d" in self.rpe_type:
            cope_2d = self.cope_2d(query, attn_logits)
            rpe_list.append(cope_2d)  # B, nH, Wh*Ww, Wh*Ww

        if len(rpe_list) > 1:
            rpe = torch.stack(rpe_list, dim=-1).to(
                rpe_list[0].device
            )  # B, nH, Wh*Ww, Wh*Ww, len(rpe_list)
            rpe = self.rpe_comb(rpe).squeeze(-1)  # B, nH, Wh*Ww, Wh*Ww
            return rpe

        else:
            return rpe_list[0]

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))
        ).exp()
        attn = attn * logit_scale

        # relative position bias
        rpe = self.get_rpe(B_, q, attn)
        attn = attn + rpe

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int | tuple[int]): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: LayerNorm
        pretrained_window_size (int | tuple[int]): Window size in pre-training.
        rpe_type (str): combination of relative position enconding to use, separated by ';'. Eg. 'rpe_default; cope_1d'.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        pretrained_window_size=0,
        rpe_type="rpe_default",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=pretrained_window_size,
            rpe_type=[text.strip() for text in rpe_type.split(";")],
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Input height and width.
        in_c (int): Number of input channels.
        out_c (int | None): Number of output channels. Defaults to in_c if None.
        conv_config (dict): Additional convolution config
        conv_dw (bool): Whether to use depthwise conv
        drop_path (float): Drop path regularization
        use_act_block (bool): Whether or not to use activation block
        use_ln (bool): Wheter or not to use layer norm
        ln_before (bool): Whether to use layer_norm before or after conv
        resid (bool): Whether or not to use residual convs
    """

    def __init__(
        self,
        input_resolution,
        in_c,
        out_c=None,
        conv_config={
            "kernel_size": 2,
            "stride": 2,
        },
        conv_dw=False,
        drop_path=0.0,
        use_act_block=False,
        use_ln=True,
        ln_before=False,
        resid=True,
    ):
        super().__init__()

        if conv_dw:
            conv_config["groups"] = in_c

        self.resid = resid

        self.first_block = nn.ModuleList()

        out_c = out_c or in_c

        first_conv = nn.Conv2d(in_c, out_c, **conv_config)
        self.first_block.append(first_conv)

        if use_ln:
            if ln_before:
                # LayerNorm for conv: data is channels first
                ln = LayerNorm(in_c, eps=1e-6, data_format="channels_first")
                self.first_block.insert(0, ln)

            else:
                ln = LayerNorm(out_c, eps=1e-6, data_format="channels_first")
                self.first_block.append(ln)

        self.act_block = (
            nn.Sequential(
                nn.Linear(
                    out_c, 4 * out_c
                ),  # pointwise/1x1 convs, implemented with linear layers
                nn.GELU(),
                GRN(4 * out_c),
                nn.Linear(4 * out_c, out_c),
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
            )
            if use_act_block
            else None
        )

        # Check whether to recompute resid after first block or not
        self.resid_after_first_block = (
            does_conv2d_change_dim(
                input_resolution,
                first_conv,
            )
            and self.resid
        )

        self.input_resolution = input_resolution
        self.output_resolution = compute_conv2d_output_resolution_from_module(
            input_resolution=input_resolution,
            conv_module=first_conv,
        )
        self.n_patches = self.output_resolution[0] * self.output_resolution[1]

    def forward(self, x, channels_last=True):
        """
        x: B, H, W, C (B, C, H, W if channels_last=False)
        """

        # If channels_last is True, permute to False for conv
        if channels_last:
            x = x.permute(0, 3, 1, 2)

        resid = x if self.resid else 0

        for layer in self.first_block:
            x = layer(x)

        if self.resid_after_first_block:
            resid = x

        if self.act_block is not None:
            x = self.act_block(x)

        x += resid

        # Convert to channels_last
        x = x.permute(0, 2, 3, 1)

        return x


class BasicLayer(nn.Module):
    r"""A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int | tuple[int]): Local window size.
        out_dim (int | None, optional): Number of output channels (after patch merging). Default: None
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: LayerNorm
        apply_downsample (bool): Apply downsample layer at the end of the layer. Default: False
        downsample_config (dict | None, optional): Config for downsample layer
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int | tuple[int]): Local window size in pre-training.
        rpe_type (str): combination of relative position enconding to use, separated by ';'. Eg. 'rpe_default; cope_1d'.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        out_dim=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNorm,
        apply_downsample=False,
        downsample_config=None,
        use_checkpoint=False,
        pretrained_window_size=0,
        rpe_type="rpe_default",
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                    rpe_type=rpe_type,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if apply_downsample:
            self.downsample = PatchMerging(
                input_resolution=input_resolution,
                in_c=dim,
                out_c=out_dim,  # Defaults to in_c if None
                **downsample_config,
            )
        else:
            self.downsample = None

        self.output_resolution = (
            self.downsample.output_resolution if self.downsample else None
        )
        self.n_patches = self.downsample.n_patches if self.downsample else None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = seq_to_img(x, self.input_resolution)

            # channels_last is always true for the model, except possibly at patch_embed
            x = self.downsample(x, channels_last=True)

            x = img_to_seq(x)
        return x

    def _init_respostnorm(self):
        # By initializing both the weights and biases of these layers to 0,
        # this function essentially disables the effect of the normalization
        # layers in the early stages of training.

        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class HiTNeXt(nn.Module):
    r"""Hierarchical Transformer NeXt

    Args:
        img_size (int | tuple[int]): Input image size. Default 224
        in_chans (int): Number of input image channels. Default: 3
        channels_last (bool): Whether channel dim is the last or first. Default: False
        n_stages (int): Number of Swin Transformer stages.
        embed_dim (int | tuple[int]): Patch embedding dimension.
        depths (int | tuple[int]): Depth of each Swin Transformer layer.
        num_heads (int | tuple[int]): Number of attention heads in different layers.
        window_size (int | tuple[int]): Window size. Default: 7
        mlp_ratio (float | tuple[float]): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool | tuple[bool]): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float| tuple[float]): Dropout rate. Default: 0
        attn_drop_rate (float| tuple[float]): Attention dropout rate. Default: 0
        drop_path_rate (float| tuple[float]): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        rpe_type (str): combination of relative position enconding to use, separated by ';'. Eg. 'rpe_default; cope_1d'. Default: 'rpe_default'.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_size (tuple[int]): Pretrained window sizes of each layer.
        apply_out_head (bool | None, optional): Whether to apply linear head to the output. Default: False
        out_head_dim (int | None): Out head dimension. Default: None
        patch_embed_config (dict): Patch Merging block config for patch embedding
        patch_merge_config (dict | tuple[dict]): Patch Merging block config for patch merging
    """

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        channels_last=False,
        n_stages=4,
        embed_dim=(96, 192, 384, 768),
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=LayerNorm,
        ape=False,
        rpe_type="rpe_default",
        use_checkpoint=False,
        pretrained_window_size=7,
        apply_out_head=False,
        out_head_dim=None,
        patch_embed_config={},
        patch_merge_config={},
    ):
        super().__init__()

        self.n_stages = n_stages
        self.in_chans = in_chans
        self.channels_last = channels_last

        self.img_size = to_tuple(img_size, 2)
        self.embed_dim = to_tuple(embed_dim, n_stages)
        self.depths = to_tuple(depths, n_stages)
        self.num_heads = to_tuple(num_heads, n_stages)
        self.window_size = to_tuple(window_size, n_stages)
        self.mlp_ratio = to_tuple(mlp_ratio, n_stages)
        self.qkv_bias = to_tuple(qkv_bias, n_stages)
        self.drop_rate = to_tuple(drop_rate, n_stages)
        self.attn_drop_rate = to_tuple(attn_drop_rate, n_stages)
        self.drop_path_rate = to_tuple(drop_path_rate, n_stages)
        self.pretrained_window_size = to_tuple(pretrained_window_size, n_stages)
        self.patch_merge_config = to_tuple(patch_merge_config, n_stages - 1)
        self.rpe_type = to_tuple(rpe_type, n_stages)

        self.patch_embed_config = patch_embed_config
        self.norm_layer = norm_layer
        self.ape = ape
        self.use_checkpoint = use_checkpoint
        self.apply_out_head = apply_out_head
        self.out_head_dim = out_head_dim

        self.patch_embed = PatchMerging(
            input_resolution=self.img_size,
            in_c=self.in_chans,
            out_c=self.embed_dim[0],
            **self.patch_embed_config,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.n_patches, self.embed_dim[0])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        self.patches_resolution = [self.patch_embed.output_resolution]

        self.layers = nn.ModuleList()
        for i_layer in range(self.n_stages):
            layer = BasicLayer(
                dim=self.embed_dim[i_layer],
                out_dim=(
                    self.embed_dim[i_layer + 1]
                    if (i_layer < self.n_stages - 1)
                    else None
                ),
                input_resolution=self.patches_resolution[i_layer],
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size[i_layer],
                mlp_ratio=self.mlp_ratio[i_layer],
                qkv_bias=self.qkv_bias[i_layer],
                drop=self.drop_rate[i_layer],
                attn_drop=self.attn_drop_rate[i_layer],
                drop_path=dpr[
                    sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                apply_downsample=True if (i_layer < self.n_stages - 1) else False,
                downsample_config=(
                    self.patch_merge_config[i_layer]
                    if (i_layer < self.n_stages - 1)
                    else None
                ),
                use_checkpoint=self.use_checkpoint,
                pretrained_window_size=self.pretrained_window_size[i_layer],
                rpe_type=self.rpe_type[i_layer],
            )
            self.layers.append(layer)

            if layer.output_resolution is not None:
                self.patches_resolution.append(layer.output_resolution)

        if self.apply_out_head:
            self.norm = norm_layer(self.embed_dim[-1])
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.head = (
                nn.Linear(self.embed_dim[-1], self.out_head_dim)
                if (self.out_head_dim is not None) and (self.out_head_dim > 0)
                else nn.Identity()
            )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    torch.jit.ignore

    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", "relative_position_bias_table"}

    def forward_features(self, x):
        x = self.patch_embed(x, channels_last=self.channels_last)
        x = img_to_seq(x)
        # After patch embed, channels_last is always True

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.apply_out_head:
            x = self.norm(x)  # B L C
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
            x = self.head(x)

        return x
