import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import math


class LocallyConnected2d(nn.Module):

    def __init__(
        self,
        input_size,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.bias = bias

        in_height, in_width = _pair(input_size)
        self.out_height = int(
            math.floor(
                (
                    in_height
                    + 2 * self.padding[0]
                    - self.dilation[0] * (self.kernel_size[0] - 1)
                    - 1
                )
                / self.stride[0]
                + 1
            )
        )
        self.out_width = int(
            math.floor(
                (
                    in_width
                    + 2 * self.padding[1]
                    - self.dilation[1] * (self.kernel_size[1] - 1)
                    - 1
                )
                / self.stride[1]
                + 1
            )
        )

        self.weight = nn.Parameter(
            torch.Tensor(
                self.out_height,
                self.out_width,
                self.out_channels,
                self.in_channels,
                *self.kernel_size,
            )
        )

        if self.bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels, self.out_height, self.out_width)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.bias is None:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)

        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def _conv_forward(self, input, weight, bias=None):
        return conv2d_local(
            input, weight, bias, self.stride, self.padding, self.dilation
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)


def conv2d_local(input, weight, bias, stride=1, padding=0, dilation=1):
    if input.dim() != 4:
        raise NotImplementedError(
            "Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim())
        )

    if weight.dim() != 6:
        raise NotImplementedError(
            "Input Error: Only 6D weight Tensors supported (got {}D)".format(
                weight.dim()
            )
        )
    outH, outW, outC, inC, kH, kW = weight.size()

    unfold = nn.Unfold(
        (kH, kW),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    cols = unfold(input)  # B x [inC * kH * kW] x [outH * outW]
    B = cols.size(0)
    cols = cols.view(B, inC * kH * kW, outH * outW, 1).permute(0, 2, 3, 1)

    output = torch.matmul(
        cols,
        weight.view(
            outH * outW,
            outC,
            inC * kH * kW,
        ).permute(0, 2, 1),
    )
    output = output.view(B, outH, outW, outC).permute(0, 3, 1, 2)

    if bias is not None:
        output = output + bias.expand_as(output)

    return output
