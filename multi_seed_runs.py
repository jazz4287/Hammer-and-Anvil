from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.strategy_args import StrategyArgs
import copy
import os
from train import parse_args

from run_existing_attacks import clip_params, train


def grid(aggregator_args: AggregatorArgs,
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
    original_model_file = env_args.model_file
    # seeds = [1,2]  # 3 runs total but we already ran seed = 0
    seeds = [0,1,2]
    for seed in seeds:
        # set_seed(env_args.seed)
        # setting parameters
        env_args.seed = seed
        if aggregator_args.aggregator_name == "mean" and aggregator_args.mean_type == "emp":
            defense = "none"
        elif aggregator_args.aggregator_name == "krum" and aggregator_args.krum_type == "top" and aggregator_args.m_krum == 1:
            defense = "krum"
        elif aggregator_args.aggregator_name == "norm" and aggregator_args.norm_type == "clip":
            defense = "norm"
        elif aggregator_args.aggregator_name == "mean" and aggregator_args.mean_type == "median_of_means":
            defense = "mom"
        elif aggregator_args.aggregator_name == "mean" and aggregator_args.mean_type == "rob_median_of_means":
            defense = "rob_mom"
        else:
            raise NotImplementedError
        assert hasattr(aggregator_args, "fine_tune_clip")
        setattr(aggregator_args, "fine_tune_clip", clip_params[defense][aggregator_args.fine_tune])
        # Run whatever you want from here. Most likely initializing a strategy and running it with the loaded args.
        # it might look something like this

        # set_tf32(env_args.use_tf32)
        print(f"#################")
        print(f"Starting grid for aggregator: {aggregator_args.aggregator_name}, {aggregator_args.mean_type}")
        print(f"#################")

        # if env_args.enable_onednn:
        #     enable_onednn_fusion()
        original_client_args = copy.deepcopy(client_args)
        original_env_args = copy.deepcopy(env_args)
        if strategy_args.strategy_name == "replacement":
            malicious_clients_list = [1,]  # the number of malicious clients does not really matter here
        else:
            malicious_clients_list = [1, 2, 4, 8]
        if backdoor_args.backdoor_name == "dba":
            malicious_clients_list = [4,]
        # lrs = [0.3, 0.03, 0.003, 0.0003]
        if env_args.dataset == "cifar10":
            lrs = [0.01, ]
        elif env_args.dataset == "mnist":
            lrs = [0.02, ]
        else:
            raise NotImplementedError

        for malicious_clients in malicious_clients_list:
            for lr in lrs:
                client_args = copy.deepcopy(original_client_args)
                env_args = copy.deepcopy(original_env_args)
                client_args.benign_lr = lr
                client_args.num_malicious_clients = malicious_clients
                mal_lr_str = ""  # to avoid floating shenanigans
                if (aggregator_args.aggregator_name == "mean" and strategy_args.strategy_name != "replacement" and
                        strategy_args.strategy_name != "fmn"):
                    if lr == 0.3:
                        client_args.malicious_lr = lr * 5
                        mal_lr_str = "5"
                    else:
                        client_args.malicious_lr = lr*10
                        mal_lr_str = str(client_args.benign_lr).split('.')[1]
                        if mal_lr_str[-2] == "0":
                            mal_lr_str = mal_lr_str[:-2] + mal_lr_str[-1]  # remove a 0
                else:
                    client_args.malicious_lr = lr
                    mal_lr_str = str(client_args.malicious_lr).split('.')[1]

                if backdoor_args.attack_type == "dba":
                    # special case, we re-use the paper's parameters to remain as faithful as possible to their attack.
                    client_args.malicious_lr = lr/2
                    client_args.malicious_epoch = client_args.benign_epoch * 3  # taken from their paper
                    mal_lr_str = str(client_args.malicious_lr).split('.')[1]

                base_name = f"model_{client_args.num_clients}_{client_args.num_malicious_clients}_{str(client_args.benign_lr).split('.')[1]}_{mal_lr_str}_{aggregator_args.fine_tune}"
                if aggregator_args.aggregator_name == "norm":
                    model_name = base_name + f"_{aggregator_args.norm_bound}.pth"
                elif aggregator_args.aggregator_name == "krum":
                    model_name = base_name + ".pth"
                elif aggregator_args.aggregator_name == "mean" and (aggregator_args.mean_type == "median_of_means" or aggregator_args.mean_type == "rob_median_of_means"):
                    model_name = base_name + f"_{aggregator_args.mom_blocks}.pth"
                else:
                    model_name = base_name + ".pth"
                # env_args.model_file = model_name
                # adding the seed to the name
                env_args.model_file = os.path.splitext(model_name)[0] + f"_{seed}" + \
                                      os.path.splitext(model_name)[1]

                train(aggregator_args, backdoor_args, strategy_args, client_args, env_args)


if __name__ == "__main__":
    grid(*parse_args())
