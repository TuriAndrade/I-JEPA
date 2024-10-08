import torch

# B = 10
# nH = 4
# N = 6
# C = 8
# wH = 2
# wW = 3

# x = torch.ones(B, nH, N, N)

# print("x slice:", x[3, 2, :, :])

# gates_h = x.transpose(-1, -2).reshape(
#     B,
#     nH,
#     N,
#     wW,
#     wH,
# )

# print("gates_h slice:", gates_h[3, 2, :, :, :])

# gates_w = x.reshape(
#     B,
#     nH,
#     N,
#     wH,
#     wW,
# )

# print("gates_w slice:", gates_w[3, 2, :, :, :])

# pos_h = gates_h.flip(-1).cumsum(dim=-1).flip(-1)
# pos_w = gates_w.flip(-1).cumsum(dim=-1).flip(-1)

# pos_h = pos_h.reshape(
#     B,
#     nH,
#     N,
#     N,
# ).transpose(-1, -2)

# print("pos_h slice", pos_h[3, 2, :, :])

# pos_w = pos_w.reshape(
#     B,
#     nH,
#     N,
#     N,
# )

# print("pos_w slice", pos_w[3, 2, :, :])

# # sum positions (ensure unique sums)
# pos_h *= wW
# pos = pos_h + pos_w

# print("pos_h sum slice", pos_h[3, 2, :, :])

# print("final pos:", pos[3, 2, :, :])

# import torch.nn as nn
# from models.utils import (
#     compute_conv2d_output_shape_from_module,
#     does_conv2d_change_dim,
#     compute_conv2d_output_fraction,
# )

# # Define a convolutional layer
# conv_layer = nn.Conv2d(
#     in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=0, dilation=1
# )

# # Example input shape (height, width)
# input_shape = (256, 128)
# print("in shape", input_shape)

# # Compute output shape
# output_shape = compute_conv2d_output_shape_from_module(input_shape, conv_layer)
# print("out shape", output_shape)

# print("dim change", does_conv2d_change_dim(input_shape, conv_layer))

# print("fraction", compute_conv2d_output_fraction(input_shape, conv_layer))

import torch
from models import HiTNeXt  # Import your HiTNeXt model here


def test_hitnext():
    # Set up the model
    model = HiTNeXt(
        img_size=224,  # Input image size
        in_chans=3,  # Number of input channels (e.g., RGB = 3)
        channels_last=False,  # Whether channel dimension is last
        n_stages=4,  # Number of Swin Transformer stages
        embed_dim=(96, 192, 384, 768),  # Embedding dimensions
        depths=(2, 2, 6, 2),  # Depth of each stage
        num_heads=(3, 6, 12, 24),  # Number of attention heads
        window_size=7,  # Attention window size
        mlp_ratio=4.0,  # MLP hidden dimension ratio
        qkv_bias=True,  # Whether to use bias in QKV
        drop_rate=0.0,  # Dropout rate
        attn_drop_rate=0.0,  # Attention dropout rate
        drop_path_rate=0.1,  # Stochastic depth
        ape=False,  # Absolute position embedding
        rpe_type="cope_2d",  # Relative position encoding type
        apply_out_head=True,  # Apply a linear head to the output
        out_head_dim=1000,  # Output head dimension (e.g., classification)
    )

    # Define a random input tensor with shape (batch_size, channels, height, width)
    batch_size = 2
    input_tensor = torch.randn(
        batch_size, 3, 224, 224
    )  # Example input (RGB image of size 224x224)

    # Set model to evaluation mode
    model.eval()

    # Run the input tensor through the model
    with torch.no_grad():  # No need to track gradients for testing
        output = model(input_tensor)

    # Print the output shape
    print(f"Output shape: {output.shape}")  # Should be (batch_size, out_head_dim)


if __name__ == "__main__":
    test_hitnext()
