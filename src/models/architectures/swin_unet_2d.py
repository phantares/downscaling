import torch
import torch.nn as nn
from .swin_transformer_2d import BasicLayer, PatchMerging, UpSample


class SwinUnet2D(nn.Module):
    def __init__(
        self,
        img_size=[112, 96],
        patch_size=[4, 4],
        in_chans=1,
        embed_dim=96,
        depths=[2, 6, 6, 2],
        num_heads=[2, 6, 6, 2],
        window_size=[2, 2],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.swintransformer_C = [0, 1, 1, 0]
        self.swintransformer_dim = [0, 1, 1, 0]
        self.window_size = window_size

        self.patch_embedded = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )
        self.patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) // 2)
        ] + [x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths) // 2)]
        self.layers = nn.ModuleList()
        for i_layer in range(4):
            layer = BasicLayer(
                dim=int(embed_dim * (2 ** self.swintransformer_dim[i_layer])),
                input_resolution=(
                    self.patches_resolution[0]
                    // (2 ** self.swintransformer_C[i_layer]),
                    self.patches_resolution[1]
                    // (2 ** self.swintransformer_C[i_layer]),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
            )
            self.layers.append(layer)

        self.downsample = PatchMerging(
            self.patches_resolution, dim=embed_dim, norm_layer=norm_layer
        )
        self.upsample = UpSample(
            (self.patches_resolution[0] // 2, self.patches_resolution[1] // 2),
            embed_dim * 2,
            norm_layer=norm_layer,
        )

        self.patch_recovery = nn.ConvTranspose2d(
            embed_dim * 2, 1, kernel_size=patch_size, stride=patch_size, padding=0
        )
        self.avg_pool = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1, count_include_pad=False
        )

    def forward(self, input_surface):
        x = self.patch_embedded(input_surface)
        x = x.flatten(2).transpose(1, 2)
        x = self.layers[0](x)
        skip_con = x

        x = self.downsample(x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.upsample(x)
        x = self.layers[3](x)

        x = torch.concat([skip_con, x], dim=-1)
        # patch recovery
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(
            x.shape[0],
            x.shape[1],
            self.patches_resolution[0],
            self.patches_resolution[1],
        )
        x = self.patch_recovery(x)
        x = self.avg_pool(x)

        return x
