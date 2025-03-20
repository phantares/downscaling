import torch
from .frameworks import FrameworkType


class ModelBuilder:

    def __init__(self, model_configs) -> None:
        self.model = FrameworkType[model_configs.framework.name].value(model_configs)

        if "pre_ckpt" in model_configs.architecture:
            self._load_pretrained_weight(model_configs.architecture.pre_ckpt)

    def get_model(self):
        return self.model

    def _load_pretrained_weight(self, pretrained_ckpt: str):
        pre_weight = torch.load(pretrained_ckpt)["state_dict"]
        new_weight = {}

        for layer in pre_weight.keys():
            new_layer = layer.replace("model.", "generator.")
            new_weight[new_layer] = pre_weight[layer]

        self.model.load_state_dict(new_weight, strict=False)
