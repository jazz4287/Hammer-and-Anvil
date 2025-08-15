import torch
import tqdm

from src.arguments.client_args import ClientArgs
from src.arguments.env_args import EnvArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.backdoors.backdoor_factory import BackdoorFactory
from src.clients.malicious_client import MaliciousClient
from src.datasets.dataset import Dataset
from src.utils.logging import Logger
from src.datasets.data_distributor import DataDistributor


class StaticBackdoorClient(MaliciousClient):
    def __init__(
        self,
        client_args: ClientArgs,
        backdoor_args: BackdoorArgs,
        env_args: EnvArgs = None,
        logger: Logger = None,
    ):
        super(StaticBackdoorClient, self).__init__(client_args, backdoor_args, env_args, logger)

    def train_one_epoch(self, model_state_dict, benign_states, *args, **kwargs):
        if self.backdoor_args.refresh_backdoor:
            assert self.original_dataset is not None, f"Original dataset is None, it means the dataset was not set before training"
            self.set_dataset(self.original_dataset.copy())

        if kwargs.get("no_backdoor", None) is not None and kwargs["no_backdoor"] == True:
            self.set_dataset(self.original_dataset.copy(), no_backdoor=True)

        self._setup_fit(model_state_dict)

        for i in range(self.client_args.malicious_epoch):
            self.progress_bar = tqdm.tqdm(total=len(self.dataloader), desc="Steps", position=0, leave=True)
            self._local_eval()
            self._epoch_fit()
            self._local_eval()
        
        self._log_epoch(self.clean_test_accuracy, self.loss, metadata={}, backdoor_accuracy=self.backdoor_asr)

        model = self.model.state_dict()
        for k in model.keys():
            # do this because mal_epoch can be different from benign_epoch
            # want all model updates to share the same num_batches_tracked
            # this is not a problem when mal_epoch == benign_epoch
            if "num_batches_tracked" in k:
                model[k] = model_state_dict[k] + self.client_args.benign_epoch * len(self.dataloader)
                # model[k] = model_state_dict[k] * (self.client_args.malicious_epoch / self.client_args.benign_epoch)
        if kwargs.get("no_backdoor", None) is not None and kwargs["no_backdoor"] == True:
            # we reset back to using the backdoor
            self.set_dataset(self.original_dataset.copy(), no_backdoor=False)

        return model
