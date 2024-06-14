import torch
import torch.nn as nn
from timm.models.layers import DropPath


class SwinTransformerLayer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        dim: int,
        heads: int,
        depth: int,
        window_shape: tuple[int],
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ) -> None:

        super().__init__()

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    input_shape,
                    dim,
                    heads,
                    window_shape,
                    roll=(i % 2 == 1),
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.blocks:
            x = layer(x)

        return x


class SwinTransformerBlock(nn.Module):
    """3D Swin Transformer Block with EarthSpecificBias.

    Args:
        input_shape (tuple[int]): Shape of input tensor in (Z, H, W). The input tensor represents a 3D image after patch embedding.
        dim (int): Number of input channels.
        heads (int): Number of attention heads.
        window_shape (tuple[int]): Window shape in (wZ, wH, wW).
        roll (bool): Whether to use shifted window attention.
        drop (float, optional): Output dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        input_shape: tuple[int],
        dim: int,
        heads: int,
        window_shape: tuple[int],
        roll: bool,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ) -> None:

        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = EarthAttention3D(
            input_shape,
            dim,
            heads,
            window_shape,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MultilayerPerceptron(dim, dropout_rate=drop)

        self.input_shape = input_shape
        self.dim = dim
        self.heads = heads
        self.window_shape = window_shape
        self.roll = roll

        if roll:
            self.shift_shape = tuple(i // 2 for i in self.window_shape)
            attention_mask = self._generate_3d_attention_mask()
        else:
            attention_mask = None

        self.register_buffer("attention_mask", attention_mask)

    def _generate_3d_attention_mask(self) -> torch.Tensor:
        img_mask = torch.zeros(
            (1, self.input_shape[0], self.input_shape[1], self.input_shape[2], 1)
        )

        z_slices = (
            slice(0, -self.window_shape[0]),
            slice(-self.window_shape[0], -self.shift_shape[0]),
            slice(-self.shift_shape[0], None),
        )
        h_slices = (
            slice(0, -self.window_shape[1]),
            slice(-self.window_shape[1], -self.shift_shape[1]),
            slice(-self.shift_shape[1], None),
        )
        w_slices = (
            slice(0, -self.window_shape[2]),
            slice(-self.window_shape[2], -self.shift_shape[2]),
            slice(-self.shift_shape[2], None),
        )

        cnt = 0
        for z in z_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, z, h, w, :] = cnt
                    cnt += 1

        mask_windows = partition_window(img_mask, self.window_shape)
        mask_windows = mask_windows.reshape(
            -1, self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
        )

        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0, float(-100.0)
        ).masked_fill(attention_mask == 0, float(0.0))

        return attention_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z, H, W = self.input_shape
        B, N, C = x.shape
        assert N == Z * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, Z, H, W, C)

        if self.roll:
            x = x.roll(shifts=[-i for i in self.shift_shape], dims=(1, 2, 3))

        x = partition_window(x, self.window_shape)  # (B*num_windows, wZ, wH, wW, dim)
        x = x.reshape(
            -1, self.window_shape[0] * self.window_shape[1] * self.window_shape[2], C
        )

        x = self.attn(x, mask=self.attention_mask)

        x = reverse_window(x, self.window_shape, self.input_shape)

        if self.roll:
            x = x.roll(shifts=self.shift_shape, dims=(1, 2, 3))

        x = x.reshape(B, H * W * Z, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class EarthAttention3D(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        input_shape (tuple[int]): Shape of input tensor in (Z, H, W).
        dim (int): Number of input channels.
        heads (int): Number of attention heads.
        window_shape (tuple[int]): Window shape in (wZ, wH, wW).
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        proj_drop (float, optional): Output dropout rate. Default: 0.0
    """

    def __init__(
        self,
        input_shape: tuple[int],
        dim: int,
        heads: int,
        window_shape: tuple[int],
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:

        super().__init__()

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.window_size = window_shape

        self.wZ, self.wH, self.wW = window_shape
        self.nZ, self.nH, self.nW = map(lambda x, y: x // y, input_shape, window_shape)
        self.window_types = self.nZ * self.nH

        self.earth_specific_bias = torch.zeros(
            ((self.wZ**2 * self.wH**2 * (2 * self.wW - 1)), self.window_types, heads)
        )
        self.earth_specific_bias = nn.Parameter(self.earth_specific_bias)
        self.earth_specific_bias = nn.init.trunc_normal_(
            self.earth_specific_bias, std=0.02
        )

        self.register_buffer("position_index", self._construct_index())

    def _construct_index(self) -> torch.Tensor:
        coords_zi = torch.arange(self.wZ)
        coords_zj = -torch.arange(self.wZ) * self.wZ
        coords_hi = torch.arange(self.wH)
        coords_hj = -torch.arange(self.wH) * self.wH
        coords_w = torch.arange(self.wW)

        coords_1 = torch.stack(
            torch.meshgrid(coords_zi, coords_hi, coords_w, indexing="ij"), dim=0
        )
        coords_2 = torch.stack(
            torch.meshgrid(coords_zj, coords_hj, coords_w, indexing="ij"), dim=0
        )
        coords_1_flatten = torch.flatten(coords_1, start_dim=1)
        coords_2_flatten = torch.flatten(coords_2, start_dim=1)
        coords = coords_1_flatten[:, :, None] - coords_2_flatten[:, None, :]
        coords = coords.permute(1, 2, 0)

        coords[:, :, 2] += self.wW - 1
        coords[:, :, 1] *= 2 * self.wW - 1
        coords[:, :, 0] *= (2 * self.wW - 1) * self.wH**2

        position_index = coords.sum(-1)
        position_index = position_index.flatten()

        return position_index

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, wZ*wH*wW, dim)
            mask: (0/-inf) mask with shape of (num_windows, wH*wW, wH*wW) or None
        """
        B_, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.heads, self.dim // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        query = query * self.scale
        attention = query @ key.transpose(
            -2, -1
        )  # (B*num_windows, heads, wZ*wH*wW, wZ*wH*wW)

        EarthSpeicificBias = self.earth_specific_bias[self.position_index]
        EarthSpeicificBias.reshape(
            (
                self.wZ * self.wH * self.wW,
                self.wZ * self.wH * self.wW,
                self.window_types,
                self.heads,
            )
        ).permute(
            2, 3, 0, 1
        )  # (num_windows, heads, wZ*wH*wW, wZ*wH*wW)

        attention = attention + EarthSpeicificBias

        if mask is not None:
            attention = attention.reshape(
                B_ // self.window_types, self.window_types, self.heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.reshape(-1, self.heads, N, N)

        attention = self.softmax(attention)
        attention = self.attn_drop(attention)

        x = (attention @ value).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MultilayerPerceptron(nn.Module):
    def __init__(self, dim: int, dropout_rate: float) -> None:
        super().__init__()

        self.linear1 = nn.Linear(dim, dim * 4)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)

        return x


def partition_window(
    x: torch.Tensor, window_shape: tuple[int, int, int]
) -> torch.Tensor:
    """
    Args:
        x: (B, Z, H, W, C)
        window_shape (list): (wZ, wH, wW)
    Returns:
        windows: (B*num_windows, wZ, wH, wW, C)
    """
    B, Z, H, W, C = x.shape
    wZ, wH, wW = window_shape

    x = x.reshape(B, Z // wZ, wZ, H // wH, wH, W // wW, wW, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, wZ, wH, wW, C)

    return windows


def reverse_window(
    windows: torch.Tensor, window_shape: tuple[int], original_shape: tuple[int]
) -> torch.Tensor:
    """
    Args:
        windows: (B*num_windows, wZ, wH, wW, C)
        window_shape (tuple[int, int, int]): (wZ, wH, wW)
        original_shape (tuple[int, int, int]): (Z, H, W)

    Returns:
        x: (B, Z, H, W, C)
    """

    wZ, wH, wW = window_shape
    nZ, nH, nW = map(lambda x, y: x // y, original_shape, window_shape)

    x = windows.reshape(-1, nZ, nH, nW, wZ, wH, wW, windows.shape[-1])
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(
        -1, wZ * nZ, wH * nH, wW * nW, windows.shape[-1]
    )

    return x
