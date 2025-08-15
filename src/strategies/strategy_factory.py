from src.arguments.strategy_args import StrategyArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.backdoor_args import BackdoorArgs

# from src.utils.logging import Logger

from src.strategies.fed_avg_strategy import FedAvgStrategy
from src.strategies.no_fed_strategy import NoFedStrategy
from src.strategies.krum_opt_strategy import KrumOptStrategy
from src.strategies.norm_opt_strategy import NormOptStrategy
from src.strategies.mom_opt_strategy import MoMOptStrategy
# from src.strategies.ga_strategy import GAStrategy
from src.strategies.model_replacement_strategy import ModelReplacementStrategy
from src.strategies.forget_me_not_strategy import FMNStrategy
from src.strategies.dba_strategy import DBAStrategy

class StrategyFactory:

    strategies = {
        "fed_avg": FedAvgStrategy,
        "no_fed": NoFedStrategy,
        "krum_opt": KrumOptStrategy,
        "norm_opt": NormOptStrategy,
        "mom_opt": MoMOptStrategy,
        "replacement": ModelReplacementStrategy,
        "fmn": FMNStrategy,
        "dba": DBAStrategy
        # "ga": GAStrategy
    }

    @classmethod
    def from_strategy_args(
        cls,
        strategy_args: StrategyArgs,
        client_args: ClientArgs,
        aggregator_args: AggregatorArgs,
        backdoor_args: BackdoorArgs,
        env_args: EnvArgs = None,
    ):
        strategy = cls.strategies.get(strategy_args.strategy_name, None)
        if strategy is None:
            raise ValueError(strategy_args.strategy_name)
        return strategy(
            strategy_args,
            client_args,
            aggregator_args,
            backdoor_args,
            env_args,
        )
