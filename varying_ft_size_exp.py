from run_existing_attacks import defense_params, clip_params, parse_args, set_run_params, train
from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.strategy_args import StrategyArgs
import os


fine_tune_sizes_to_try = [100, 500, 1000, 1500, 2000, 5000]
base_save_path: str = "./saved_models/varying_ft_size_exp/"


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
    seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        print(f"##########\nRunning seed: {seed}\n##########")
        env_args.seed = seed
        for fine_tune_size in fine_tune_sizes_to_try:
            assert hasattr(aggregator_args, "fine_tune")
            setattr(aggregator_args, "fine_tune", fine_tune_size)
            if strategy_args.strategy_name == "fed_avg":
                defense = "none"
            elif strategy_args.strategy_name == "krum_opt":
                defense = "krum"
            else:
                raise NotImplementedError
            defense_param = defense_params[defense]
            save_path = os.path.join(base_save_path, f"fine_tune_{fine_tune_size}", defense)
            # now we set the params for the defense
            for param, param_value in defense_param["aggregator_args"].items():
                assert hasattr(aggregator_args, param)
                setattr(aggregator_args, param, param_value)

            # set the clip value for fine-tuning
            # however some clip values are not in the dictionary. If <500, we use 500 clip val, if >2000, we use 2000 clip val
            # in the middle we interpolate
            assert hasattr(aggregator_args, "fine_tune_clip")
            if fine_tune_size not in list(clip_params[defense].keys()):
                if fine_tune_size < 500:
                    clip_value = clip_params[defense][500]
                elif fine_tune_size > 2000:
                    clip_value = clip_params[defense][2000]
                else:
                    clip_value = clip_params[defense][500] - (clip_params[defense][500] - clip_params[defense][2000])*((fine_tune_size - 500)/ (2000 - 500))
                    assert clip_value > 0
                    if clip_params[defense][500] >= clip_params[defense][2000]:
                        assert clip_params[defense][500] >= clip_value >= clip_params[defense][2000]
                    else:
                        assert clip_params[defense][500] < clip_value < clip_params[defense][2000]
                setattr(aggregator_args, "fine_tune_clip", clip_value)
            else:
                setattr(aggregator_args, "fine_tune_clip", clip_params[defense][fine_tune_size])

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
            print(f"Running fine_tune {fine_tune_size} {defense}")
            # adding the seed to the name
            env_args.model_file = os.path.splitext(model_name)[0] + f"_{seed}" + \
                                  os.path.splitext(model_name)[1]
            train(aggregator_args, backdoor_args, strategy_args, client_args, env_args)

if __name__ == "__main__":
    main(*parse_args())