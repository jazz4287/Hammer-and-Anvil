from src.arguments.client_args import ClientArgs
from src.arguments.env_args import EnvArgs
from src.models.base_models.resnet import *
from src.models.base_models.lenet import LeNet5
from src.models.base_models.cnn import CNN


class ModelFactory:
    models = {
        "lenet": LeNet5,
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152,
        "cnn": CNN,
    }

    @staticmethod
    def build_special_args(model_name: str, env_args: EnvArgs):
        kwargs = {}
        # TODO: models?
        # if model_name.startswith("lenet") or 
        if model_name.startswith("resnet") or model_name == "cnn":
            kwargs["input_channels"] = env_args.get_num_input_channels()
            kwargs["num_classes"] = env_args.get_num_classes()
        return kwargs

    @classmethod
    def from_client_args(cls, client_args: ClientArgs, env_args: EnvArgs):
        if (client_args is None) or (env_args is None):
            raise ValueError("client_args and env_args must be provided")

        model = cls.models.get(client_args.model_name, None)
        if model is None:
            raise ValueError(client_args.model_name)

        kwargs = ModelFactory.build_special_args(client_args.model_name, env_args)

        return model(**kwargs)
