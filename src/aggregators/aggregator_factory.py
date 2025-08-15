from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.aggregators.base_aggregators.mean_aggregator import MeanAggregator
from src.aggregators.base_aggregators.krum_aggregator import KrumAggregator
from src.aggregators.base_aggregators.norm_aggregator import NormAggregator


class AggregatorFactory:

    aggregators = {"mean": MeanAggregator, "krum": KrumAggregator, "norm": NormAggregator}

    @classmethod
    def from_aggregator_args(
        cls,
        aggregator_args: AggregatorArgs,
        client_args: ClientArgs,
        env_args: EnvArgs = None,
    ):
        aggregator = cls.aggregators.get(aggregator_args.aggregator_name, None)
        if aggregator is None:
            raise ValueError(aggregator_args.aggregator_name)
        return aggregator(aggregator_args, client_args, env_args)
