from dataclasses import dataclass, field


@dataclass
class ClientArgs:
    CONFIG_KEY = "client_args"
    """ This class contains all arguments for the clients. """

    model_name: str = field(
        default="resnet18",
        metadata={
            "help": "name of the classification model to use",
            "choices": ["cnn", "resnet18", "resnet50", "lenet"],
        },
    )

    optimizer: str = field(
        default="sgd",
        metadata={
            "choices": ["adam", "sgd"],
            "help": "which optimizer should the clients use to train their local version of the model"
        },
    )

    num_clients: int = field(
        default=5,
        metadata={"help": "number of clients for the federated learning process"},
    )

    num_malicious_clients: int = field(
        default=0,
        metadata={
            "help": "number of clients from the total number of clients (num_clients) that are malicious"
        },
    )

    subset_size: int = field(
        default=1,
        metadata={
            "help": "number of clients to include in each iteration"
        }
    )

    # lr: float = field(
    #     default=0.001,
    #     metadata={
    #         "help": "Learning rate to be used by the local models to train their local copy of the model."
    #     },
    # )

    benign_epoch: int = field(
        default = 1,
        metadata={
            "help": "number of epochs benign client runs when selected"
        }
    )

    benign_lr: float = field(
        default=0.001,
        metadata={
            "help": "learning rate for benign client"
        }
    )

    malicious_epoch: int = field(
        default=1,
        metadata={
            "help": "number of epochs malicious client runs when selected"
        }
    )

    malicious_lr: float = field(
        default=0.001,
        metadata={
            "help": "learning rate for malicious client"
        }
    )
    
    # "only_aggregator" not valid, nothing can be done w/o benign updates
    malicious_level: str = field(
        default="none",
        metadata={
            "help": "how much info about benign clients and aggregator does malicious client have",
            "choices": ["none", "only_benign_clients", "benign_clients_and_aggregator"]
        }
    )

    # for dba
    dba_poison_start_epoch: int = field(default=0)

    dba_poison_end_epoch: int = field(default=-1)

    dba_single_shot: bool = field(default=False)
    dba_client_0_epoch: int = field(default=-1)
    dba_client_1_epoch: int = field(default=-1)
    dba_client_2_epoch: int = field(default=-1)
    dba_client_3_epoch: int = field(default=-1)

    dba_param_scaler: float = field(default=1.0)

    # neurotoxin
    neurotoxin_mask_ratio: float = field(default=0.95)  # 0.95 means only the top 5% are masked and it's what they use the most in their paper
