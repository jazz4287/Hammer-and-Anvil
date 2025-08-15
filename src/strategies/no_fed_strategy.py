import torch
from os import path

from src.strategies.strategy import Strategy
from src.arguments.strategy_args import StrategyArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.backdoor_args import BackdoorArgs

from src.clients.client_factory import ClientFactory
from src.datasets.data_distributor import DataDistributor
from src.models.model_factory import ModelFactory
from src.utils.models import get_new_save_folder, test_model
from src.utils.gpu import DEVICE


class NoFedStrategy(Strategy):
    def __init__(
        self,
        strategy_args: StrategyArgs,
        client_args: ClientArgs,
        aggregator_args: AggregatorArgs,
        backdoor_args: BackdoorArgs,
        env_args: EnvArgs = None,
    ):
        super(NoFedStrategy, self).__init__(
            strategy_args, client_args, aggregator_args, backdoor_args, env_args
        )

    def run(self):
        print("No Federated Strategy.")
        save_folder = get_new_save_folder("no_fed", save_path=self.env_args.save_path)
        device = torch.device(DEVICE)

        # initialize model to train
        model = ModelFactory().from_client_args(self.client_args, self.env_args)
        model.to(device)

        # get data
        dist = DataDistributor(self.client_args, self.env_args)
        master_loader = dist.get_eval_dataloader()

        client = ClientFactory.from_client_args(
            self.client_args, client_type="benign", env_args=self.env_args
        )

        for epoch in range(self.strategy_args.num_epochs):
            print(f"Epoch {epoch}: Starting training.")

            client.set_dataloader(master_loader)
            new_dict = client.train_one_epoch(
                model.state_dict(),
            )
            model.load_state_dict(new_dict)
            model.to(device)

            print(f"Epoch {epoch}: Training complete.")

            test_accuracy = test_model(model, master_loader, device)
            print(f"Test Accuracy: {test_accuracy}.")

            # TODO: move to utils with "custom descriptor"
            # save model
            # save_path = path.join(save_folder, f"no_fed_{epoch}.pt")
            # torch.save(model.state_dict(), save_path)
            # print(f"Saved model to {save_folder}.")
