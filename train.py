import transformers
import torch

from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.strategy_args import StrategyArgs
from src.strategies.strategy_factory import StrategyFactory
from src.utils.models import set_seed
from src.utils.gpu import enable_onednn_fusion, set_tf32
import os


def parse_args():
    parser = transformers.HfArgumentParser(
        (AggregatorArgs, BackdoorArgs, StrategyArgs, ClientArgs, EnvArgs, ConfigArgs)
    )
    return parser.parse_args_into_dataclasses()


def train(
    aggregator_args: AggregatorArgs,
    backdoor_args: BackdoorArgs,
    strategy_args: StrategyArgs,
    client_args: ClientArgs,
    env_args: EnvArgs,
    config_args: ConfigArgs,
):
    if (
        config_args.exists()
    ):  # a configuration file was provided. yaml files always overwrite other settings!
        client_args = (
            config_args.get_client_args()
        )  # params to instantiate the generator.
        env_args = config_args.get_env_args()
        aggregator_args = config_args.get_aggregator_args()
        backdoor_args = config_args.get_backdoor_args()
        strategy_args = config_args.get_strategy_args()

    # Run whatever you want from here. Most likely initializing a strategy and running it with the loaded args.
    # it might look something like this
    set_seed(env_args.seed)

    set_tf32(env_args.use_tf32)

    if env_args.enable_onednn:
        enable_onednn_fusion()

    strategy = StrategyFactory.from_strategy_args(
        strategy_args,
        client_args=client_args,
        aggregator_args=aggregator_args,
        backdoor_args=backdoor_args,
        env_args=env_args,
    )
    strategy.run()
    
    model = strategy.aggregator.model
    save_file =  os.path.join(env_args.save_path, env_args.model_file)
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")

    if aggregator_args.fine_tune > 0:
        strategy.fine_tune()


if __name__ == "__main__":
    train(*parse_args())
