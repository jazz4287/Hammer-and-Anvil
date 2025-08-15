from dataclasses import dataclass, field
import yaml

from src.arguments.env_args import EnvArgs
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.strategy_args import StrategyArgs
from src.arguments.client_args import ClientArgs


@dataclass
class ConfigArgs:
    config_path: str = field(
        default=None, metadata={"help": "path to the yaml configuration file (*.yml)"}
    )

    def exists(self):
        return self.config_path is not None

    args_to_config = {  # specify these keys in the *.yml file
        AggregatorArgs.CONFIG_KEY: AggregatorArgs,
        EnvArgs.CONFIG_KEY: EnvArgs,
        ClientArgs.CONFIG_KEY: ClientArgs,
        BackdoorArgs.CONFIG_KEY: BackdoorArgs,
        StrategyArgs.CONFIG_KEY: StrategyArgs,
    }

    def get_env_args(self) -> EnvArgs:
        return self.loaded_configs.setdefault(EnvArgs.CONFIG_KEY, EnvArgs())

    def get_client_args(self) -> ClientArgs:
        return self.loaded_configs.setdefault(ClientArgs.CONFIG_KEY, ClientArgs())

    def get_aggregator_args(self) -> AggregatorArgs:
        return self.loaded_configs.setdefault(
            AggregatorArgs.CONFIG_KEY, AggregatorArgs()
        )

    def get_backdoor_args(self) -> BackdoorArgs:
        return self.loaded_configs.setdefault(BackdoorArgs.CONFIG_KEY, BackdoorArgs())

    def get_strategy_args(self) -> StrategyArgs:
        return self.loaded_configs.setdefault(StrategyArgs.CONFIG_KEY, StrategyArgs())

    def __post_init__(self):
        self.load_config()

    def load_config(self):
        """
        Load from config file to dataclass
        """
        if self.config_path is None:
            print("> No config file specified. Using default values.")
            return
        self.loaded_configs = {}

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)
        self.keys = list(data.keys())  # all keys specified in the yaml

        for entry in data.keys():
            cls = self.args_to_config[entry]
            values = {}
            if hasattr(
                cls, "from_checkpoint"
            ):  # load from a local or remote checkpoint
                values = cls.from_checkpoint(**values)
            values.update(data[entry])  # yaml always overrides everything
            if hasattr(cls, "load_specific_class"):  # composability pattern
                cls = cls.load_specific_class(**values)
            self.loaded_configs[entry] = cls(**values)
