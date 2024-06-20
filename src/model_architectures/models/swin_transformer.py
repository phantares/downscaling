import torch
import torch.nn as nn
from timm.models.layers import DropPath
from einops import repeat


class SwinTransformerLayer(nn.Module):
    """
    Basic layer of Swin Transformer, contains even numbers of blocks.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        window_shape: tuple[int, int, int],
        dim: int,
        heads: int,
        depth: int,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: list[float] = [0.0],
    ) -> None:
        """
        Args:
            input_shape (tuple[int, int, int]): Shape of input after patch embedding (Z, H, W).
            window_shape (tuple[int, int, int]): Shape of window (wZ, wH, wW).
            dim (int): Number of input channels.
            heads (int): Number of attention heads.
            roll (bool): Whether to use shifted window attention.
            drop (float, optional): Output dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
        """

        super().__init__()

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    input_shape=input_shape,
                    window_shape=window_shape,
                    dim=dim,
                    heads=heads,
                    roll=(i % 2 == 1),
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i].item(),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, Z*H*W, dim).
        Returns:
            x (torch.Tensor): (B, Z*H*W, dim).
        """

        for block in self.blocks:
            x = block(x)

        return x


class SwinTransformerBlock(nn.Module):
    """
    3D Swin Transformer Block with EarthSpecificBias.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        window_shape: tuple[int, int, int],
        dim: int,
        heads: int,
        roll: bool,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        """
        Args:
            input_shape (tuple[int, int, int]): Shape of input after patch embedding (Z, H, W).
            window_shape (tuple[int, int, int]): Shape of window (wZ, wH, wW).
            dim (int): Number of input channels.
            heads (int): Number of attention heads.
            roll (bool): Whether to use shifted window attention.
            drop (float, optional): Output dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
        """

        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.attn = EarthAttention3D(
            input_shape=input_shape,
            window_shape=window_shape,
            dim=dim,
            heads=heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MultilayerPerceptron(dim=dim, dropout_rate=drop)

        self.Z, self.H, self.W = input_shape
        self.wZ, self.wH, self.wW = window_shape
        self.dim = dim
        self.heads = heads
        self.roll = roll

        if roll:
            self.shift_shape = tuple(i // 2 for i in window_shape)
            attention_mask = self._generate_3d_attention_mask()
        else:
            attention_mask = None

        self.register_buffer("attention_mask", attention_mask)

    def _generate_3d_attention_mask(self) -> torch.Tensor:
        """
        Returns:
            attention_mask (torch.Tensor): (num_windows, wZ*wH*wW, wZ*wH*wW)
        """

        img_mask = torch.zeros((1, self.Z, self.H, self.W, 1))

        z_slices = (
            slice(0, -self.wZ),
            slice(-self.wZ, -self.shift_shape[0]),
            slice(-self.shift_shape[0], None),
        )
        h_slices = (
            slice(0, -self.wH),
            slice(-self.wH, -self.shift_shape[1]),
            slice(-self.shift_shape[1], None),
        )
        w_slices = (
            slice(0, -self.wW),
            slice(-self.wW, -self.shift_shape[2]),
            slice(-self.shift_shape[2], None),
        )

        cnt = 0
        for z in z_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, z, h, w, :] = cnt
                    cnt += 1

        mask_windows = partition_window(
            img_mask, window_shape=(self.wZ, self.wH, self.wW)
        )
        mask_windows = mask_windows.reshape(-1, self.wZ * self.wH * self.wW)

        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0, float(-100.0)
        ).masked_fill(attention_mask == 0, float(0.0))

        return attention_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, Z*H*W, dim).
        Returns:
            x (torch.Tensor): (B, Z*H*W, dim).
        """

        B, L, C = x.shape
        assert L == self.Z * self.H * self.W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, self.Z, self.H, self.W, C)

        if self.roll:
            x = x.roll(shifts=[-i for i in self.shift_shape], dims=(1, 2, 3))

        x = partition_window(
            x, window_shape=(self.wZ, self.wH, self.wW)
        )  # (B*num_windows, wZ, wH, wW, dim)
        x = x.reshape(
            -1, self.wZ * self.wH * self.wW, C
        )  # (B*num_windows, wZ*wH*wW, dim)

        x = self.attn(x, mask=self.attention_mask)

        x = reverse_window(
            x,
            window_shape=(self.wZ, self.wH, self.wW),
            original_shape=(self.Z, self.H, self.W),
        )  # (B, Z, H, W, dim)

        if self.roll:
            x = x.roll(shifts=self.shift_shape, dims=(1, 2, 3))

        x = x.reshape(B, self.Z * self.H * self.W, C)  # (B, Z*H*W, dim)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class EarthAttention3D(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with Earth specific bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        window_shape: tuple[int, int, int],
        dim: int,
        heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            input_shape (tuple[int, int, int]): Shape of input after patch embedding (Z, H, W).
            window_shape (tuple[int, int, int]): Shape of window in (wZ, wH, wW).
            dim (int): Number of input channels.
            heads (int): Number of attention heads.
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            proj_drop (float, optional): Output dropout rate. Default: 0.0
        """

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
        """
        Returns:
            position_index (torch.Tensor)
        """

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
            x (torch.Tensor): (B*num_windows, wZ*wH*wW, dim)
            mask (torch.Tensor): (0/-inf) mask in (num_windows, wZ*wH*wW, wZ*wH*wW) or None
        Returns:
            x (torch.Tensor): (B*num_windows, wZ*wH*wW, dim)
        """

        B_, L, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B_, L, 3, self.heads, self.dim // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # (B*num_windows, heads, wZ*wH*wW, dim//heads)

        query = query * self.scale
        attention = query @ key.transpose(
            -2, -1
        )  # (B*num_windows, heads, wZ*wH*wW, wZ*wH*wW)

        EarthSpecificBias = self.earth_specific_bias[
            self.position_index
        ]  # (wZ*wH*wW^2, window_types, heads)
        EarthSpecificBias = EarthSpecificBias.reshape(
            (
                self.wZ * self.wH * self.wW,
                self.wZ * self.wH * self.wW,
                self.window_types,
                self.heads,
            )
        ).permute(
            2, 3, 0, 1
        )  # (window_types, heads, wZ*wH*wW, wZ*wH*wW)
        EarthSpecificBias = repeat(
            EarthSpecificBias,
            "win head zhw1 zhw2 ->  (f win) head zhw1 zhw2",
            f=B_ // self.window_types,
        )  # (B*num_windows, heads, wZ*wH*wW, wZ*wH*wW)

        attention = attention + EarthSpecificBias

        if mask is not None:
            mask = repeat(
                mask,
                "win zhw1 zhw2 ->  (b win) zhw1 zhw2",
                b=B_ // (self.window_types * self.nW),
            )  # (B*num_windows, wZ*wH*wW, wZ*wH*wW)

            attention = attention + mask[::, None]

        attention = self.softmax(attention)
        attention = self.attn_drop(attention)

        x = (attention @ value).transpose(1, 2).reshape(B_, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MultilayerPerceptron(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.0) -> None:
        """
        Args:
            dim (int): Number of input channels.
            dropout_rate (float, optional): Output dropout rate. Default: 0.0
        """

        super().__init__()

        self.linear1 = nn.Linear(dim, dim * 4)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, Z*H*W, dim).
        Returns:
            x (torch.Tensor): (B, Z*H*W, dim).
        """

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
        x (torch.Tensor): (B, Z, H, W, C)
        window_shape (tuple[int, int, int]): (wZ, wH, wW)
    Returns:
        windows (torch.Tensor): (B*num_windows, wZ, wH, wW, C)
    """

    B, Z, H, W, C = x.shape
    wZ, wH, wW = window_shape

    x = x.reshape(B, Z // wZ, wZ, H // wH, wH, W // wW, wW, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, wZ, wH, wW, C)

    return windows


def reverse_window(
    windows: torch.Tensor,
    window_shape: tuple[int, int, int],
    original_shape: tuple[int, int, int],
) -> torch.Tensor:
    """
    Args:
        windows (torch.Tensor): (B*num_windows, wZ*wH*wW, C)
        window_shape (tuple[int, int, int]): (wZ, wH, wW)
        original_shape (tuple[int, int, int]): (Z, H, W)
    Returns:
        x (torch.Tensor): (B, Z, H, W, C)
    """

    wZ, wH, wW = window_shape
    nZ, nH, nW = map(lambda x, y: x // y, original_shape, window_shape)

    x = windows.reshape(-1, nZ, nH, nW, wZ, wH, wW, windows.shape[-1])
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(
        -1, wZ * nZ, wH * nH, wW * nW, windows.shape[-1]
    )

    return x
