import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from src.datasets.data_distributor import DataDistributor
import tqdm

from src.arguments.client_args import ClientArgs
from src.arguments.env_args import EnvArgs
from src.models.model_factory import ModelFactory

from src.utils.logging import Logger, History
from src.utils.models import test_model
from src.datasets.dataset import Dataset


class Client:
    def __init__(
        self,
        client_args: ClientArgs,
        env_args: EnvArgs = None,
        logger: Logger = None,
        device=None,
    ):
        self.client_args = client_args
        self.env_args = env_args if env_args is not None else EnvArgs()

        self.is_malicious = False
        self.current_epoch = -1
        self.id = -1
        self.dataset = None
        self.dataloader = None
        self.model = ModelFactory.from_client_args(client_args, env_args=env_args)
        self.optimizer = self.get_optimizer(self.model.parameters(), self.client_args.optimizer, self.client_args.benign_lr)

        self.progress_bar = None
        self.loss = -1

        if device is None:
            # TODO:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        self.logger = logger
        if self.logger is None:
            # give warning
            print("Warning: No logger provided. Progress will not be saved.")

    def _setup_fit(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        print(f"Found {len(self.dataset)} samples.")

    def _epoch_fit(self):
        self.model.train()
        losses = 0.0
        count = 0
        for _, (image, target) in enumerate(self.dataloader):
            # yeet features and labels to gpu
            image, target = image.to(self.device), target.to(self.device)

            # forward
            output = self.model(image)
            loss = F.cross_entropy(output, target)
            losses += float(loss.item()) * image.size(0)
            count += image.size(0)
            self.optimizer.zero_grad()

            # back prop
            loss.backward()

            # update
            self.optimizer.step()

            # log progress
            self.progress_bar.update(1)
            self.progress_bar.set_description(f"Steps ({loss.item():.3f})")

        self.progress_bar.close()
        self.loss = losses / count

    def _local_eval(self):
        self.model.eval()
        self.test_accuracy = test_model(self.model, self.dataloader, self.device)

        print(f"Avg Training Loss: {self.loss:.3f} | Clean Accuracy: {self.test_accuracy}")

    def _log_epoch(self, local_test_accuracy, loss, metadata=None, backdoor_accuracy=0):
        if self.logger is not None:
            hist = History(
                train_loss=loss,
                train_accuracy=local_test_accuracy,
                backdoor_loss=0,  # TODO: backdoor
                backdoor_accuracy=backdoor_accuracy,
                metadata={},
            )
            if metadata is not None:
                hist.metadata = metadata

            self.logger.log_client(self.id, epoch=self.current_epoch, history=hist)
        else:
            print("Client: No logger set.")

    @staticmethod
    def get_optimizer(model_params, optimizer, lr):
        # TODO: allow changing optimizers
        if optimizer == "adam":
            return Adam(model_params, lr=lr)
        elif optimizer == "sgd":
            return SGD(model_params, lr=lr)
        else:
            raise NotImplementedError

    def set_model_state(self, model_state):
        self.model.load_state_dict(model_state)

    def set_id(self, client_id: int):
        self.id = client_id

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset
        self.dataloader = DataDistributor.make_dataloader(self.dataset, shuffle=True,
                                                          batch_size=self.env_args.batch_size)

    def set_logger(self, logger: Logger):
        self.logger = logger

    def train_one_epoch(self, model_state_dict, *args, **kwargs):
        """
        Takes in the global model state dictionary, loads it into the local model. Then train the local model
        with the client's dataset and optimizer for one epoch before returning the local model's state dictionary as
        output.
        :param model_state_dict: state dictionary of parameters obtained from the aggregator
        :param args:
        :param kwargs:
        :return:
        """

        self._setup_fit(model_state_dict)

        for i in range(self.client_args.benign_epoch):
            self._local_eval()
            self.progress_bar = tqdm.tqdm(total=len(self.dataloader), desc="Steps", position=0, leave=True)
            self._epoch_fit()
            self.progress_bar.close()
            self._local_eval()

        self._log_epoch(self.test_accuracy, self.loss, metadata={})

        return self.model.state_dict()
