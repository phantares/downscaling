import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

class MapPlotter:

    def __init__(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        self.data = data

    def plot_map(self):
        lon, lat = self._get_lon_lat()
        colors, norm = self._get_colormap()

        fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(lon, lat, self.data, cmap=colors, norm=norm)

        return fig

    def _get_lon_lat(self):
        latStart = 21
        latEnd =26.55
        lonStart = 118.2+0.125*4
        lonEnd = 123.35-0.125*4

        lat = np.linspace(latStart,latEnd,112)
        lon = np.linspace(lonStart,lonEnd,96)
        lon, lat = np.meshgrid(lon, lat)

        return lon, lat

    def _get_colormap(self):
        colors = mpl.colors.ListedColormap(
            [
                "#FFFFFF",
                "#9CFCFF",
                "#03C8FF",
                "#059BFF",
                "#0363FF",
                "#059902",
                "#39FF03",
                "#FFFB03",
                "#FFC800",
                "#FF9500",
                "#FF0000",
                "#CC0000",
                "#990000",
                "#960099",
                "#C900CC",
                "#FB00FF",
                "#FDC9FF",
            ]
        )
        bounds = [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
        norm = mpl.colors.BoundaryNorm(bounds, colors.N)

        return colors, norm