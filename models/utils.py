import math


def to_tuple(value, n):
    if isinstance(value, (tuple, list)):
        if len(value) != n:
            raise ValueError(f"Expected length {n}, but got length {len(value)}.")
        return value
    else:
        return (value,) * n


def compute_conv2d_output_resolution_from_params(
    input_resolution,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    **kwargs,
):
    r"""
    Compute the output shape for a 2D convolution operation in PyTorch.

    Parameters:
    - input_resolution (int | tuple[int]): A tuple (height, width) representing the input resolution.
    - kernel_size (int or tuple): Size of the convolving kernel.
    - stride (int or tuple, optional): Stride of the convolution. Default is 1.
    - padding (int or tuple, optional): Implicit zero padding on both sides of the input. Default is 0.
    - dilation (int or tuple, optional): Spacing between kernel elements. Default is 1.

    Returns:
    - output_resolution (tuple): A tuple (out_height, out_width) representing the output resolution.
    """

    input_resolution = to_tuple(input_resolution, 2)
    kernel_size = to_tuple(kernel_size, 2)
    stride = to_tuple(stride, 2)
    padding = to_tuple(padding, 2)
    dilation = to_tuple(dilation, 2)

    in_height, in_width = input_resolution

    out_height = (
        math.floor(
            (in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
            / stride[0]
        )
        + 1
    )

    out_width = (
        math.floor(
            (in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
            / stride[1]
        )
        + 1
    )

    return out_height, out_width


import math


def compute_conv2d_output_resolution_from_module(input_resolution, conv_module):
    r"""
    Compute the output shape for a 2D convolution operation using a PyTorch Conv2d module.

    Parameters:
    - input_resolution (int | tuple[int]): A tuple (height, width) representing the input resolution.
    - conv_module (nn.Conv2d): A PyTorch Conv2d module.

    Returns:
    - output_resolution (tuple): A tuple (out_height, out_width) representing the output resolution.
    """
    kernel_size = conv_module.kernel_size
    stride = conv_module.stride
    padding = conv_module.padding
    dilation = conv_module.dilation

    out_height, out_width = compute_conv2d_output_resolution_from_params(
        input_resolution=input_resolution,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    return out_height, out_width


def does_conv2d_change_dim(input_resolution, conv_module):
    r"""
    Compute the output shape for a 2D convolution operation using a PyTorch Conv2d module.

    Parameters:
    - input_resolution (int | tuple[int]): A tuple (height, width) representing the input resolution.
    - conv_module (nn.Conv2d): A PyTorch Conv2d module.

    Returns:
    - dim_changed (tuple[bool]): A tuple (height_changed, width_changed) indicating if the output resolution changed.
    """
    in_height, in_width = to_tuple(input_resolution, 2)
    out_height, out_width = compute_conv2d_output_resolution_from_module(
        input_resolution=input_resolution,
        conv_module=conv_module,
    )

    return in_height != out_height, in_width != out_width


def compute_conv2d_output_fraction_from_module(input_resolution, conv_module):
    r"""
    Compute the fraction of the output dimensions compared to the input dimensions
    for a 2D convolution operation.

    Parameters:
    - input_resolution (int | tuple[int]): A tuple (height, width) representing the input resolution.
    - conv_module (nn.Conv2d): A PyTorch Conv2d module.

    Returns:
    - output_fraction (tuple): A tuple (height_fraction, width_fraction) representing
      the fraction of output dimensions to input dimensions.
    """

    in_height, in_width = to_tuple(input_resolution, 2)
    out_height, out_width = compute_conv2d_output_resolution_from_module(
        input_resolution, conv_module
    )

    height_fraction = out_height / in_height
    width_fraction = out_width / in_width

    return height_fraction, width_fraction
