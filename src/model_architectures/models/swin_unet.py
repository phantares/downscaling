import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformerLayer


class SwinUnet(nn.Module):
    """
    3D SwinUnet similar to Pangu-Weather.
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        patch_shape: tuple[int, int, int],
        window_shape: tuple[int, int, int],
        embed_dim: int,
        heads: list[int],
        depths: list[int],
        surface_channels: int,
        upper_channels: int = 0,
        upper_levels: int = 0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        interpolation: bool = False,
        **avg_pool_configs,
    ) -> None:
        """
        Args:
            image_shape (tuple[int, int]): Shape of input image (imgH, imgW).
            patch_shape (tuple[int, int, int]): Shape of patch (pZ, pH, pW).
            window_shape (tuple[int, int, int]): Shape of window (wZ, wH, wW).
            embed_dim (int): Dimension of patch embedding.
            heads (list[int]): Number of attention heads in each layer.
            depths (list[int]): Number of swin transformer blocks in each layer.
            surface_channels (int): Channels of surface variables.
            upper_channels (int, optional): Channels of upper variables. Default: 0
            upper_levels (int, optional): Number of levels in upper tensor. Default: 0
            drop_rate (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path_rate (float, optional): Maximum of stochastic depth rate. Default: 0.2
            avg_pool_configs: Settings for AvgPool2d.
        """

        super().__init__()

        num_layers = len(depths)
        image_shape = [upper_levels] + image_shape
        dpr = torch.linspace(0, drop_path_rate, sum(depths))

        Z, H, W = map(lambda x, y: x // y, image_shape, patch_shape)
        Z += 1  # surface

        self.shape = image_shape
        self.interp = interpolation

        self.patch_embed = PatchEmbedding(
            patch_shape=patch_shape,
            dim=embed_dim,
            surface_channels=surface_channels,
            upper_channels=upper_channels,
        )

        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            layer = SwinTransformerLayer(
                input_shape=(Z, H // (2**i), W // (2**i)),
                dim=embed_dim * (2**i),
                heads=heads[i],
                depth=depth,
                window_shape=window_shape,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
            )

            self.layers.append(layer)

        self.downsample = DownSample(input_shape=(Z, H, W), dim=embed_dim)

        self.upsample = UpSample(output_shape=(Z, H, W), dim=embed_dim)

        for i, depth in enumerate(reversed(depths), start=1):
            j = num_layers - i
            layer = SwinTransformerLayer(
                input_shape=(Z, H // (2**j), W // (2**j)),
                dim=embed_dim * (2**j),
                heads=heads[j],
                depth=depth,
                window_shape=window_shape,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:j]) : sum(depths[: j + 1])],
            )
            self.layers.append(layer)

        self.patch_recover = PatchRecovery(
            input_shape=(Z, H, W),
            patch_shape=patch_shape[1:],
            dim=embed_dim * 2,
        )

        self.activation = nn.ReLU()

        self.avg_pool = nn.AvgPool2d(**avg_pool_configs, count_include_pad=False)

    def forward(
        self,
        input_surface: torch.Tensor,
        input_upper: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_surface (torch.Tensor): (B, C_surface, imgH, imgW).
            input_upper (torch.Tensor, optional): (B, C_upper, imgZ, imgH, imgW).
        Returns:
            x (torch.Tensor): (B, 1, imgH, imgW)
        """

        if self.interp:
            input_surface = F.interpolate(input_surface, size=self.shape[1:])
            if input_upper:
                input_upper = F.interpolate(input_upper, size=self.shape)

        x = self.patch_embed(input_surface, input_upper)
        x = self.layers[0](x)
        skip = x
        x = self.downsample(x)
        x = self.layers[1](x)

        x = self.layers[2](x)
        x = self.upsample(x)
        x = self.layers[3](x)
        x = torch.cat([skip, x], dim=-1)
        x = self.patch_recover(x)
        x = self.activation(x)
        x = self.avg_pool(x)

        return x


class PatchEmbedding(nn.Module):
    """
    Convert input fields to patches and lineraly embed them.
    """

    def __init__(
        self,
        patch_shape: tuple[int, int, int],
        dim: int,
        surface_channels: int,
        upper_channels: int = 0,
    ) -> None:
        """
        Args:
            patch_shape (tuple[int, int, int]): Shape of patch (pZ, pH, pW).
            dim (int): Dimension of output embedding.
            surface_channels (int): Channels of surface variables.
            upper_channels (int, optional): Channels of upper variables. Default: 0
        """

        super().__init__()

        self.conv_surface = nn.Conv2d(
            in_channels=surface_channels,
            out_channels=dim,
            kernel_size=patch_shape[1:],
            stride=patch_shape[1:],
        )

        if upper_channels > 0:
            self.conv_upper = nn.Conv3d(
                in_channels=upper_channels,
                out_channels=dim,
                kernel_size=patch_shape,
                stride=patch_shape,
            )

    def forward(
        self,
        input_surface: torch.Tensor,
        input_upper: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_surface (torch.Tensor): (B, C_surface, imgH, imgW).
            input_upper (torch.Tensor, optional): (B, C_upper, imgZ, imgH, imgW).
        Returns:
            x (torch.Tensor): (B, Z*H*W, dim).
        """

        embedding_surface = self.conv_surface(input_surface)  # (B, dim, H, W)
        x = embedding_surface.unsqueeze(-3)  # (B, dim, 1, H, W)

        if input_upper:
            embedding_upper = self.conv_upper(input_upper)  # (B, dim, Z-1, H, W)
            x = torch.cat([embedding_upper, x], -3)  # (B, dim, Z, H, W)

        x = x.permute(0, 2, 3, 4, 1)  # (B, Z, H, W, dim)
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, Z*H*W, dim)

        return x


class PatchRecovery(nn.Module):
    """
    Recover output fields to 1 surface channel from patches.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        patch_shape: tuple[int, int],
        dim: int,
    ) -> None:
        """

        Args:
            input_shape (tuple[int, int, int]): Shape of input after patch embedding (Z, H, W).
            patch_shape (tuple[int, int]): Shape of patch (pH, pW).
            dim (int): Numbers of input channels.
        """

        super().__init__()

        self.Z, self.H, self.W = input_shape

        self.deconv = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=1,
            kernel_size=patch_shape,
            stride=patch_shape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, Z*H*W, dim).
        Returns:
            x (torch.Tensor): (B, 1, imgH, imgW).
        """

        B, L, C = x.shape

        x = x.permute(0, 2, 1)
        x = x.reshape(B, C, self.Z, self.H, self.W)
        x = self.deconv(
            x[
                :,
                :,
                -1,
            ]
        )

        return x


class DownSample(nn.Module):
    """
    Patch merging which reduces the horizontal resolution by a factor of 2.
    """

    def __init__(self, input_shape: tuple[int, int, int], dim: int) -> None:
        """
        Args:
            input_shape (tuple[int, int, int]): (Z, H, W).
            dim (int): Number of input channels.
        """

        super().__init__()

        self.Z, self.H, self.W = input_shape

        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, Z*H*W, dim).
        Returns:
            x (torch.Tensor): (B, Z * H//2 * W//2, 2*dim).
        """

        B, L, C = x.shape

        x = x.reshape(B, self.Z, self.H, self.W, C)
        x = x.reshape(B, self.Z, self.H // 2, 2, self.W // 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class UpSample(nn.Module):
    """
    Increases the horizontal resolution by a factor of 2.
    """

    def __init__(self, output_shape: tuple[int, int, int], dim: int):
        """
        Args:
            output_shape (tuple[int, int, int]): (Z, H, W).
            dim (int): Number of output channels.
        """

        super().__init__()

        self.Z, self.H, self.W = output_shape
        self.dim = dim

        self.linear1 = nn.Linear(2 * dim, 4 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, Z * H//2 * W//2,  2*dim).
        Returns:
            x (torch.Tensor): (B, Z*H*W, dim).
        """

        B, L, C = x.shape

        x = self.linear1(x)
        x = x.reshape(B, self.Z, self.H // 2, self.W // 2, 2, 2, self.dim)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, self.Z, self.H, self.W, self.dim)
        x = x.reshape(B, -1, self.dim)

        x = self.norm(x)
        x = self.linear2(x)

        return x
