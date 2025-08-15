from dataclasses import dataclass, field


@dataclass
class AggregatorArgs:
    CONFIG_KEY = "aggregator_args"
    """ This class contains all parameters for the aggregator. """

    aggregator_name: str = field(
        default="mean",
        metadata={
            "help": "name of the aggregator algorithm to use",
            "choices": ["mean", "krum", "norm"],  # currently only stable diffusion2 is supported
        },
    )

    mean_type: str = field(
        default="emp",
        metadata={
            "help": "which kind of mean aggregation to use if the mean aggregator is used",
            "choices": ["emp", "median_of_means", "rob_median_of_means"],
        },
    )

    mom_k: int = field(
        default=3,
        metadata={
            "help": "tunable param for efficient MoM bounds"
        }
    )

    mom_blocks: int = field(
        default=0,
        metadata={
            "help": "number of blocks (not to be confused with number of elements per block)"
        }
    )

    krum_type: str = field(
        default="",
        metadata={
            "help": "how to choose multiple clients when m > 1",
            # top: calculate scores for all clients, then select top m clients
            # iterative: drop top scoring client then recalculate scores for remaining m-1 clients
            "choices": ["top", "iterative"],
        },
    )

    m_krum: int = field(
        default=1,
        metadata={
            "help": "number of krum iterations (m > 0) if krum aggregator is used"
        },
    )

    norm_type: str = field(
        default="",
        metadata={
            "help": "what to do with updates that exceed norm bound",
            "choices": ["clip", "drop"]
        }
    )
    
    norm_bound: float = field(
        default = 10,
        metadata={
            "help": "max l2 norm allowed"
        }
    )

    fine_tune: int = field(
        default = 0,
        metadata={
            "help": "how many samples to use for fine-tuning, if 0, then fine-tuning is turned off."
        }
    )

    fine_tune_batch_size: int = field(
        default=32
    )

    fine_tune_epochs: int = field(
        default = 2
    )

    fine_tune_lr: float = field(
        default = 0.0001
    )

    fine_tune_clip: float = field(default=-1.)

    fine_tune_optimizer: str = field(
        default="sgd",
        metadata={
            "choices": ["adam", "sgd"],
            "help": "which optimizer should the clients use to train their local version of the model"
        },
    )