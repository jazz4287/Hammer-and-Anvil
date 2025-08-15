from dataclasses import dataclass, field
from global_settings import CACHE_DIR
# TODO: move to /tmp or some other temperatory directory (hopefully in same repo)
# CACHE_DIR = "./.cache"


@dataclass
class EnvArgs:
    CONFIG_KEY = "env_args"
    """ This class contains all arguments for the environment where to load samples and datasets. """

    save_path: str = field(
        default="./models",
        metadata={"help": "path to save the model and other artifacts"},
    )

    model_file: str = field(
        default="model.pt",
        metadata={"help": "filename for saving the trained model"}
    )

    seed: int = field(
        default=0,
        metadata={
            "help": "seed for deterministic results, used in pytorch, numpy, etc"
        },
    )

    log_interval: int = field(
        default=10,
        metadata={
            "help": "how many batches/steps to wait before logging training status or do tests"
        },
    )

    batch_size: int = field(
        default=64, metadata={"help": "default batch size for training"}
    )

    eval_batch_size: int = field(
        default=512, metadata={"help": "default batch size for evaluation"}
    )

    dataset: str = field(
        default="cifar10",
        metadata={"help": "dataset to train on", "choices": ["cifar10", "mnist"]},
    )

    use_dirichlet: bool = field(default=False, metadata={"help": "whether to use dirichlet distribution for client data"})

    dirichlet_alpha: float = field(default=0.5)

    disable_transforms: bool = field(
        default=False,
    )

    disable_normalization: bool = field(
        default=False)

    use_tf32: bool = field(
        default=False, metadata={"help": "whether to use tf32 or not"}
    )

    enable_onednn: bool = field(
        default=False, metadata={"help": "whether to use onednn optimizations"}
    )

    ga: bool = field(
        default=False,
        metadata={"help": "use GA to optimize malicious updates"}
    )

    ga_dynamic_alpha: bool = field(
        default=False,
        metadata={"help": "dynamic change alpha wrt FL iter count"}
    )

    ga_alpha: float = field(
        default=0.5,
        metadata={"help": "fitness scaling factor for backdoor accuracy"}
    )

    ga_generation: int = field(
        default=50,
        metadata={"help": "number of GA iterations"}
    )

    ga_population: int = field(
        default=20,
        metadata={"help": "number of individuals in each iteration"}
    )
    
    ga_layer: bool = field(
        default=False,
        metadata={"help": "use layer coef in addition to model coef"}
    )

    ga_elite: bool = field(
        default=False,
        metadata={"help": "keep elite across FL rounds"}
    )
    
    ga_mutation_std: float = field(
        default=0.05,
        metadata={"help": "standard deviation when using normal-based random mutation"}
    )

    ga_mutation_chance: float = field(
        default=0.01,
        metadata={"help": "per gene probability that a gene will be mutated."}
    )

    save_picked_benign: bool = field(default=False, metadata={"help": "save whether Krum picked a benign update"})

    def get_num_classes(self):
        if self.dataset == "cifar10":
            return 10
        elif self.dataset == "mnist":
            return 10
        else:
            raise NotImplementedError
        
    def get_num_input_channels(self):
        if self.dataset == "cifar10":
            return 3
        elif self.dataset == "mnist":
            return 1
        else:
            raise NotImplementedError

