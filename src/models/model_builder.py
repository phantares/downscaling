from .frameworks import FrameworkType


class ModelBuilder:

    def __init__(self, model_configs) -> None:
        self.framework = FrameworkType[model_configs.framework.name].value
        self.model = self.framework(model_configs)

    def get_framework(self):
        return self.framework

    def get_model(self):
        return self.model
