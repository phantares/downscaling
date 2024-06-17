import torch
import torch.nn as nn

from .swin_transformer import SwinTransformerLayer


class SwinUnet(nn.module):
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
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ) -> None:

        super().__init__()

        num_layers = len(depths)
        image_shape = [upper_levels] + image_shape

        Z, H, W = map(lambda x, y: x // y, image_shape, patch_shape)
        Z += 1  # surface

        dpr = torch.linspace(0, drop_path_rate, sum(depths))

        self.patch_embed = PatchEmbedding(
            patch_shape,
            dim=embed_dim,
            surface_channels=surface_channels,
            upper_channels=upper_channels,
        )

        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            layer = (
                SwinTransformerLayer(
                    input_shape=(Z, H // (2**i), W // (2**i)),
                    dim=embed_dim * (2**i),
                    heads=heads[i],
                    depth=depth,
                    window_shape=window_shape,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                ),
            )
            self.layers.append(layer)

        self.downsample = DownSample(input_shape=(Z, H, W), dim=embed_dim)

        self.upsample = UpSample(output_shape=(Z, H, W), dim=embed_dim)

        for i, depth in enumerate(reversed(depths), start=1):
            j = num_layers - i
            layer = (
                SwinTransformerLayer(
                    input_shape=(Z, H // (2**j), W // (2**j)),
                    dim=embed_dim * (2**j),
                    heads=heads[j],
                    depth=depth,
                    window_shape=window_shape,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:j]) : sum(depths[: j + 1])],
                ),
            )
            self.layers.append(layer)

        self.patch_recover = PatchRecovery(
            output_shape=(Z, H, W),
            patch_shape=patch_shape,
            upper_channels=upper_channels,
            surface_channels=surface_channels,
            dim=embed_dim * 2,
        )

    def forward(
        self,
        input_surface: torch.Tensor,
        input_upper: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unet structure.

        Args:
            input_upper (torch.Tensor): Tensor of shape (B, img_Z, img_H, img_W, Ch_upper).
            input_surface (torch.Tensor): Tensor of shape (B, 1, img_H, img_W, Ch_surface).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of upper-air data and surface data.
        """
        # left side
        x = self.patch_embed(input_surface, input_upper)
        x = self.layer[0](x)
        skip = x
        x = self.downsample(x)
        x = self.layer[1](x)
        # right side
        x = self.layer[2](x)
        x = self.upsample(x)
        x = self.layer[3](x)
        x = torch.cat([skip, x], dim=-1)
        output_surface, output_upper = self.patch_recover(x)

        return output_surface, output_upper


class PatchEmbedding(nn.module):
    def __init__(
        self,
        patch_shape: tuple[int],
        dim: int,
        surface_channels: int,
        upper_channels: int = 0,
    ) -> None:
        """
        convert input fields to patches and linearly embed them.

        Args:
            patch_shape (tuple[int, int, int]): Size of the patch (pZ, pH, pW).
            dim (int): Dimension of the output embedding.
            surface_channels (int): Channels of surface variables.
            upper_channels (int): Channels of upper-air variables.
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
            input_surface (torch.Tensor): Tensor of shape (B, C_surface, H, W).
            input_upper (torch.Tensor): Tensor of shape (B, C_upper, Z, H, W).
        Returns:
            torch.Tensor: Tensor of shape (batch_size, Z*H*W, dim).
        """

        embedding_surface = self.conv_surface(input_surface)  # (B, dim, H, W)
        x = embedding_surface.unsqueeze(-3)  # (B, dim, 1, H, W)

        if input_upper:
            embedding_upper = self.conv_upper(input_upper)  # (B, dim, Z-1, H, W)
            x = torch.cat([embedding_upper, x], -3)  # (B, dim, Z, H, W)

        x = x.permute(0, 2, 3, 4, 1)  # (B, Z, H, W, dim)
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, Z*H*W, dim)

        return x


class PatchRecovery(nn.module):
    def __init__(
        self,
        output_shape: tuple[int],
        patch_shape: tuple[int],
        dim: int,
        surface_channels: int,
        upper_channels: int = 0,
    ) -> None:
        """
        convert input fields to patches and linearly embed them.

        Args:
            patch_shape (tuple[int, int, int]): Size of the patch (pZ, pH, pW).
            dim (int): Dimension of the output embedding.
            surface_channels (int): Channels of surface variables.
            upper_channels (int): Channels of upper-air variables.
        """
        super().__init__()

        self.Z, self.H, self.W = output_shape
        self.upper_channels = upper_channels

        self.conv_surface = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=surface_channels,
            kernel_size=patch_shape[1:],
            stride=patch_shape[1:],
        )

        if upper_channels > 0:
            self.conv_upper = nn.ConvTranspose3d(
                in_channels=dim,
                out_channels=upper_channels,
                kernel_size=patch_shape,
                stride=patch_shape,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            torch.Tensor: Tensor of shape (batch_size, Z*H*W, dim).
        Returns:
            output_surface (torch.Tensor): Tensor of shape (B, C_surface, H, W).
            output_upper (torch.Tensor): Tensor of shape (B, C_upper, Z, H, W).
        """

        B, L, C = x.shape

        x = x.permute(0, 2, 1)
        x = x.reshape(B, self.Z, self.H, self.W, C)
        output_surface = self.conv_surface(
            x[
                :,
                :,
                -1,
            ]
        )
        output_surface = torch.squeeze(output_surface, -3)

        if self.upper_channels > 0:
            output_upper = self.conv_upper(
                x[
                    :,
                    :,
                    :-1,
                ]
            )
        else:
            output_upper = None

        return output_surface, output_upper


class DownSample(nn.module):
    def __init__(self, input_shape: tuple[int], dim: int) -> None:
        super().__init__()

        self.Z, self.H, self.W = input_shape

        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape

        x = x.reshape(B, self.Z, self.H, self.W, C)
        x = x.reshape(B, self.Z, self.H // 2, 2, self.W // 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class UpSample(nn.Module):
    def __init__(self, output_shape: tuple[int], dim: int):
        super().__init__()

        self.Z, self.H, self.W = output_shape

        self.linear1 = nn.Linear(2 * dim, 4 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, L, C = x.shape

        x = self.linear1(x)
        x = x.reshape(B, self.Z, self.H // 2, self.W // 2, 2, 2, C // 4)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, self.Z, self.H, self.W, C // 4)
        x = x.reshape(B, -1, C // 4)

        x = self.norm(x)
        x = self.linear2(x)

        return x
