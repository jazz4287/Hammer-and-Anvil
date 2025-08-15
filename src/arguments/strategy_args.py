from dataclasses import dataclass, field


@dataclass
class StrategyArgs:
    CONFIG_KEY = "strategy_args"
    """ This class contains all parameters for the strategy. """

    strategy_name: str = field(
        default="basic",
        metadata={
            "help": "name of the generation algorithm to use",
            "choices": [
                "fed_avg",
                "no_fed",
                "krum_opt",
                "norm_opt",
                "mom_opt",
                "ga"
            ],
        },
    )

    krum_attack: str = field(
        default="",
        metadata={
            "help": "how to optimize for krum attack",
            "choices": [
                "bisect_opt_alpha", # bisect [0,1] to find optimal alpha
                "fixed_alpha", # always use the same alpha
                "dynamic_alpha" # dynamically compute alpha based on radius scaling
            ]
        }
    )

    krum_fixed_alpha: float = field(default=0.5)

    krum_dynamic_radius_scaling: float = field(default=1.5, metadata={"help": "radius = radius / (num_of_ben_neighbor * dynamic_radius_scaling)"})

    num_epochs: int = field(
        default=20, metadata={"help": "number of epochs to train for."}
    )

    bisect_opt_alpha_depth: int = field(default=10, metadata={"help": "depth of bisect."})

    fmn_break_point: int = field(default=10)

    fmn_frequency: int = field(default=4)

    fmn_interleave_clean_training: bool = field(default=True)

    fmn_grad_clipping: float = field(default=-1)