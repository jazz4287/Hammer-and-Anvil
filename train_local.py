import transformers
import torch

from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.strategy_args import StrategyArgs
from src.clients.client_factory import ClientFactory
from src.datasets.data_distributor import DataDistributor
from src.models.model_factory import ModelFactory
from src.utils.models import get_new_save_folder, test_model
from src.utils.models import set_seed
from src.utils.gpu import enable_onednn_fusion, set_tf32
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from src.utils.logging import create_logger
import copy


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
        in_grid: bool = False
):
    if not in_grid:
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

    print("No Federated Strategy.")
    save_folder = get_new_save_folder(env_args.save_path)
    logger = create_logger(save_folder)
    logger.set_epoch(0)
    logger.set_total_epochs(strategy_args.num_epochs)
    logger.set_config_args(
        strategy_args,
        client_args,
        aggregator_args,
        backdoor_args,
        env_args,
    )
    # initialize model to train
    model = ModelFactory().from_client_args(client_args, env_args)
    model.to(device)

    # get data
    dist = DataDistributor(client_args, env_args, aggregator_args)
    eval_loader = dist.get_eval_dataloader()

    client = ClientFactory.from_client_args(
        client_args, client_type="benign", env_args=env_args
    )
    client.set_logger(logger)

    logger.save_progress()

    for epoch in range(strategy_args.num_epochs):
        print(f"Epoch {epoch}: Starting training.")

        client.set_dataset(dist.aggregator_set)
        new_dict = client.train_one_epoch(
            model.state_dict(),
        )
        model.load_state_dict(new_dict)
        model.to(device)

        print(f"Epoch {epoch}: Training complete.")

        test_accuracy = test_model(model, eval_loader, device)
        print(f"Test Accuracy: {test_accuracy}.")
        logger.update_epoch(epoch + 1, -1, test_accuracy, 0)
        logger.save_progress()
        # TODO: move to utils with "custom descriptor"
        # save model
        # save_path = path.join(save_folder, f"no_fed_{epoch}.pt")
        # torch.save(model.state_dict(), save_path)
        # print(f"Saved model to {save_folder}.")
    logger.set_done()  # set progress to "finished"
    save_file = os.path.join(env_args.save_path, env_args.model_file)
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")



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

    # Run whatever you want from here. Most likely initializing a strategy and running it with the loaded args.
    # it might look something like this
    set_seed(env_args.seed)

    set_tf32(env_args.use_tf32)
    print(f"#################")
    print(f"Starting grid for local fine tune set training")
    print(f"#################")

    if env_args.enable_onednn:
        enable_onednn_fusion()
    original_client_args = copy.deepcopy(client_args)
    original_env_args = copy.deepcopy(env_args)
    if env_args.dataset == "cifar10":
        lrs = [0.01, ]
    elif env_args.dataset == "mnist":
        lrs = [0.02,]
    else:
        raise NotImplementedError

    for lr in lrs:
        client_args = copy.deepcopy(original_client_args)
        env_args = copy.deepcopy(original_env_args)
        client_args.benign_lr = lr
        base_name = f"local_finetune_set_model_{client_args.num_clients}_{str(client_args.benign_lr).split('.')[1]}_{aggregator_args.fine_tune}"
        model_name = base_name + ".pth"
        env_args.model_file = model_name
        print(model_name)
        train(aggregator_args, backdoor_args, strategy_args, client_args, env_args, config_args, in_grid=True)



if __name__ == "__main__":
    # train(*parse_args())
    grid(*parse_args())