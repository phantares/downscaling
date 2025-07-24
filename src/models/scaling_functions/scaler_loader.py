import torch.nn as nn
from .scaling_method import ScalingMethod


class ScalerLoader(nn.Module):
    def __init__(self, model_configs):
        super().__init__()

        self.scale_rain = False
        self.rain_scaler = None
        self.scale_weather = False
        self.weather_scaler = None
        self.include_rain = model_configs.framework.config.get("include_rain", True)

        if "scaling" in model_configs:
            if "rain" in model_configs.scaling:
                self.scale_rain = True
                self.rain_scaler = self._set_scaler(model_configs.scaling.rain)

            if "weather" in model_configs.scaling:
                self.scale_weather = True
                self.weather_scaler = self._set_scaler(model_configs.scaling.weather)

    def _set_scaler(self, scaling_configs):
        scaling_function = ScalingMethod[scaling_configs.method].value(
            **scaling_configs.config
        )

        return scaling_function

    def scale(self, **inputs):
        if self.scale_rain:
            rain = inputs["input_surface"][
                :,
                -1,
            ]
            rain_scaled = self.rain_scaler.standardize(rain)

            inputs["input_surface"] = inputs["input_surface"].clone()
            inputs["input_surface"][
                :,
                -1,
            ] = rain_scaled

        if self.scale_weather:
            weather_idx = slice(None, -1) if self.include_rain else slice(None)
            weather = inputs["input_surface"][
                :,
                weather_idx,
            ]
            weather_scaled = self.weather_scaler.standardize(weather)

            inputs["input_surface"] = inputs["input_surface"].clone()
            inputs["input_surface"][
                :,
                weather_idx,
            ] = weather_scaled

            if "input_upper" in inputs:
                weather = inputs["input_upper"]
                weather_scaled = self.weather_scaler.standardize(weather, True)

                inputs["input_upper"] = inputs["input_upper"].clone()
                inputs["input_upper"] = weather_scaled

        return inputs
