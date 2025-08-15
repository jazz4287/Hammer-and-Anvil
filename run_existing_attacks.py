# Run existing attacks against our defenses (and no defense)
# best case for the attacker m=8
# fixed parameters for each of the defenses

base_save_path: str = "./saved_models/previous_work_comparison/"

fine_tune_sizes = [500, 2000]

run_params = {
    "client_args": {
        "model_name": "resnet18",
        "optimizer": "sgd",
        "benign_epoch": 1,
        "malicious_epoch": 1,
        "num_malicious_clients": 8,
        "num_clients": 20,
        "subset_size": 20,
        "benign_lr": 0.01,
        "malicious_lr": 0.01
    },
    "aggregator_args": {
        "fine_tune_epochs": 100,
    },
    "backdoor_args": {
        "k": 0.2
    },
    "strategy_args":{
        "num_epochs": 100,
    },
    "env_args": {
        "seed": 0,
        "batch_size": 64,
        "dataset": "cifar10",
        "use_tf32": False,
        "eval_batch_size": 10000
    }
}


defense_params = {
    "none": {
        "aggregator_args":{
            "aggregator_name": "mean",
            "mean_type": "emp",
        }
    },
    "krum": {
        "aggregator_args": {
            "aggregator_name": "krum",
            "krum_type": "top",
            "m_krum": 1,
        }
    },
    "norm":{
        "aggregator_args":{
            "aggregator_name": "norm",
            "norm_type": "clip",
            "norm_bound": 5,
        }
    },
    "mom":{
        "aggregator_args":{
            "aggregator_name": "mean",
            "mean_type": "median_of_means",
            "mom_blocks": 5,
        }
    },
    "rob_mom": {
        "aggregator_args": {
            "aggregator_name": "mean",
            "mean_type": "rob_median_of_means",
            "mom_blocks": 5,
        }
    }
}

acceptable_drops = {
    500: 0.2,
    2000: 0.05
}

clip_params = {
    "none": {
        500: 5,
        2000: 2,
    },
    "krum": {
        500:  5,
        2000:  2
    },
    "mom": {
        500 : 10,
        2000: 2,
    },
    "norm": {
        500 : 10,
        2000: 2
    },
    "rob_mom": {
        500 : 10,
        2000: 1
    }
}

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


def set_run_params(arg_list):
    for args in arg_list:
        if run_params.get(args.CONFIG_KEY, None) is not None:
            for key, value in run_params[args.CONFIG_KEY].items():
                assert hasattr(args, key), print(args.CONFIG_KEY, key, value)
                setattr(args, key, value)
    return arg_list


def train(
        aggregator_args: AggregatorArgs,
        backdoor_args: BackdoorArgs,
        strategy_args: StrategyArgs,
        client_args: ClientArgs,
        env_args: EnvArgs,
):


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
    save_file = os.path.join(env_args.save_path, env_args.model_file)
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")

    if aggregator_args.fine_tune > 0:
        strategy.fine_tune_with_graph()


def main(
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

    aggregator_args, backdoor_args, strategy_args, client_args, env_args = set_run_params([aggregator_args, backdoor_args, strategy_args, client_args, env_args])

    # now that we've set the run_params, we need to run the attack against every defense then fine_tune
    for fine_tune_size in fine_tune_sizes:
        # save_path = os.path.join(base_save_path, f"fine_tune_{fine_tune_size}")

        assert hasattr(aggregator_args, "fine_tune")
        setattr(aggregator_args, "fine_tune", fine_tune_size)

        for defense, defense_param in defense_params.items():
            save_path = os.path.join(base_save_path, f"fine_tune_{fine_tune_size}", defense)
            # now we set the params for the defense
            for param, param_value in defense_param["aggregator_args"].items():
                assert hasattr(aggregator_args, param)
                setattr(aggregator_args, param, param_value)

            # set the clip value for fine-tuning
            assert hasattr(aggregator_args, "fine_tune_clip")
            setattr(aggregator_args, "fine_tune_clip", clip_params[defense][fine_tune_size])

            # set the save_path and model name
            model_name = os.path.splitext(os.path.basename(config_args.config_path))[0] + ".pt"

            assert hasattr(env_args, "model_file")
            setattr(env_args, "model_file", model_name)

            assert hasattr(env_args, "save_path")
            setattr(env_args, "save_path", save_path)

            # we then set the learning-rate
            if (aggregator_args.aggregator_name == "mean" and strategy_args.strategy_name != "replacement" and
                    strategy_args.strategy_name != "fmn"):
                print(f"Scaling learning rate")
                client_args.malicious_lr = client_args.benign_lr * 10

            train(aggregator_args, backdoor_args, strategy_args, client_args, env_args)


if __name__ == "__main__":
    main(*parse_args())