import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["SparseWeightedConv2D"]

@torch.jit.script
def _agg_sparse_conv_forward(weight: torch.Tensor, x: torch.Tensor, kernel_size: tuple[int, int], padding: int, stride: int, in_channels: int, out_channels: int):
    # Unfold input (im2col) to create patches
    x_unfolded = F.unfold(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )  # Shape: (batch_size, in_channels * kernel_height * kernel_width, num_patches)

    # Transpose x_unfolded to match (in_features, batch_size * num_patches)
    x_unfolded = x_unfolded.permute(0, 2, 1).reshape(-1, x_unfolded.shape[1])

    # Sparse-dense matrix multiplication
    conv_result = torch.sparse.mm(weight, x_unfolded.T)  # Shape: (out_channels, batch_size * num_patches)

    # Reshape to convolutional output format
    batch_size, _, height, width = x.shape
    out_height = (height + 2 * padding - kernel_size[0]) // stride + 1
    out_width = (width + 2 * padding - kernel_size[1]) // stride + 1
    conv_result = conv_result.T.reshape(batch_size, out_height, out_width, out_channels).permute(0, 3, 1, 2)

    return conv_result


@torch.jit.script
def _batched_sparse_conv_forward(weight: torch.Tensor, x: torch.Tensor, kernel_size: tuple[int, int], padding: int, stride: int, in_channels: int, out_channels: int):

    batch_size, _, height, width = x.shape

    # Compute output dimensions
    out_height = (height + 2 * padding - kernel_size[0]) // stride + 1
    out_width = (width + 2 * padding - kernel_size[1]) // stride + 1

    # Unfold input to patches
    x_unfolded = F.unfold(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )  # Shape: (batch_size, in_channels * kernel_height * kernel_width, num_patches)

    # Reshape for sparse matrix multiplication
    # x_unfolded: (batch_size, in_features, num_patches)
    in_features = weight.size(1)
    num_patches = x_unfolded.size(2)
    x_unfolded = x_unfolded.reshape(batch_size, in_features, num_patches)

    # Preallocate output tensor
    conv_result = torch.empty(
        (batch_size, weight.size(0), num_patches), device=x.device
    )  # Shape: (batch_size, out_channels, num_patches)

    for b in range(batch_size):
        # Sparse-dense matrix multiplication
        # x_unfolded[b]: Shape (in_features, num_patches)
        conv_result[b] = torch.sparse.mm(weight, x_unfolded[b])

    # Reshape to convolutional output format
    conv_result = conv_result.view(batch_size, out_channels, out_height, out_width)

    return conv_result


class SparseWeightedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same", bias=True):
        super(SparseWeightedConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        if padding != "same": assert False
        self.padding = (kernel_size - 1) // 2

        self.register_parameter("weight", nn.Parameter(torch.empty(out_channels, in_channels * self.kernel_size[0] * self.kernel_size[1], device = "cuda").to_sparse_csr()))
    
    def forward(self, x):
        return _batched_sparse_conv_forward(self.weight, x, self.kernel_size, self.padding, self.stride, self.in_channels, self.out_channels)

"""import torch
import os
from torch.utils.cpp_extension import load

# Load the custom CUDA extension
sparse_conv_cuda = load(
    name="sparse_conv_cuda",
    sources=["SparseConv/sparse_conv.cpp"],
    extra_cuda_cflags=["-O3"],
    extra_include_paths=[os.path.join(os.environ['CUDA_HOME'], 'include')],
    verbose=True
)

# Alias the function from the compiled extension
_sparse_conv_forward = sparse_conv_cuda._sparse_conv_forward

# Create a custom autograd function in Python for ease of use
class SparseConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, x, kernel_size, padding, stride, in_channels, out_channels):
        # Save for backward
        ctx.save_for_backward(weight, x)
        ctx.kernel_size = kernel_size
        ctx.padding = padding
        ctx.stride = stride
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels

        # Call the forward function
        return _sparse_conv_forward(
            weight, x, kernel_size, padding, stride, in_channels, out_channels
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        weight, x = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        padding = ctx.padding
        stride = ctx.stride

        # Compute gradients
        grad_weight = torch.zeros_like(weight)
        grad_x = torch.zeros_like(x)
        
        # Call the backward logic (already handled in the CUDA code)
        # Not explicitly implemented here, use the existing pipeline.

        return grad_weight, grad_x, None, None, None, None, None

# Define input tensors
batch_size, in_channels, height, width = 2, 3, 32, 32
out_channels, kernel_height, kernel_width = 4, 3, 3

# Create a sparse weight tensor
dense_weight = torch.randn(out_channels, in_channels * kernel_height * kernel_width).cuda()
indices = torch.nonzero(dense_weight.abs() > 0.5, as_tuple=False).t()
values = dense_weight[indices[0], indices[1]]
weight = torch.sparse_coo_tensor(indices, values, dense_weight.size()).cuda()

# Create a dense input tensor
x = torch.randn(batch_size, in_channels, height, width).cuda()

# Convolution parameters
kernel_size = [kernel_height, kernel_width]
padding = [1, 1]  # Same padding
stride = [1, 1]

# Apply the sparse convolution
output = SparseConvFunction.apply(
    weight, x, kernel_size, padding, stride, in_channels, out_channels
)

print("Output Shape:", output.shape)

# Perform backward pass
output.sum().backward()

"""