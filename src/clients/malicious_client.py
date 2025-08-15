import torch
from torch.utils.data import TensorDataset
import tqdm
import matplotlib.pyplot as plt

from src.arguments.client_args import ClientArgs
from src.arguments.env_args import EnvArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.backdoors.backdoor_factory import BackdoorFactory
from src.clients.client import Client
from src.datasets.dataset import Dataset
from src.utils.logging import Logger
from src.datasets.data_distributor import DataDistributor
from src.utils.models import test_model


class MaliciousClient(Client):
    def __init__(
        self,
        client_args: ClientArgs,
        backdoor_args: BackdoorArgs,
        env_args: EnvArgs = None,
        logger: Logger = None,
    ):
        super(MaliciousClient, self).__init__(client_args, env_args, logger=logger)
        # overwrite optimizer w/ malicious learning rate
        self.optimizer = self.get_optimizer(self.model.parameters(), self.client_args.optimizer, self.client_args.malicious_lr)
        self.is_malicious = True
        self.backdoor_args = backdoor_args
        self.backdoor = BackdoorFactory.from_backdoor_args(self.backdoor_args, self.env_args)
        self.original_dataset = None
        self.clean_dataset = None
        self.backdoor_dataset = None

        self.triggers_idx = {}
        self.processed_images = 0
        self.skipped_triggers = 0
        self.poisoned = 0

    def _setup_backdoor(self):
        self.dataset, self.clean_dataset, self.backdoor_dataset, self.skipped_triggers, self.poisoned, self.processed_images = self.backdoor.process_dataset(self.dataset)
        print(f"Client {self.id}: Skipped {self.skipped_triggers} images. Poisoned {self.poisoned} images. Processed {self.processed_images} images.")
        # general dataloader kept for training
        self.dataloader = DataDistributor.make_dataloader(self.dataset, shuffle=True, batch_size=self.env_args.batch_size)
        # test set w/ eval transforms
        self.clean_dataset.set_train(False)
        self.backdoor_dataset.set_train(False)
        self.clean_loader_eval = DataDistributor.make_dataloader(self.clean_dataset, shuffle=False, batch_size=self.env_args.eval_batch_size)
        self.backdoor_loader_eval = DataDistributor.make_dataloader(self.backdoor_dataset, shuffle=False, batch_size=self.env_args.eval_batch_size)

    def set_dataset(self, new_dataset: Dataset, no_backdoor=False):
        if self.original_dataset is None:
            self.original_dataset = new_dataset.copy()  # in case we want to repeatedly poison
        self.dataset = new_dataset
        if not no_backdoor:
            if self.backdoor_args.k == -1:
                print(f"Full mode")
                # we use the full benign + full backdoor dataset concat dataset
                self.backdoor_args.k = 1
                self._setup_backdoor()
                # now we have the full backdoor dataset
                train_backdoor_dataset = self.dataset.copy()
                train_clean_dataset = self.original_dataset.copy()
                clean_images, clean_labels = train_clean_dataset.get_indexed_data_and_targets()
                backdoor_images, backdoor_labels = train_backdoor_dataset.get_indexed_data_and_targets()
                images = torch.cat([clean_images, backdoor_images], dim=0)
                labels = torch.cat([clean_labels, backdoor_labels], dim=0)
                self.dataset.set_dataset(torch.utils.data.TensorDataset(images, labels))
                self.clean_dataset = train_clean_dataset.copy()
                self.clean_dataset.set_train(False)
                self.backdoor_dataset = train_backdoor_dataset.copy()
                self.backdoor_dataset.set_train(False)
                self.clean_loader_eval = DataDistributor.make_dataloader(self.clean_dataset, shuffle=False,
                                                                         batch_size=self.env_args.eval_batch_size)
                self.backdoor_loader_eval = DataDistributor.make_dataloader(self.backdoor_dataset, shuffle=False,
                                                                        batch_size=self.env_args.eval_batch_size)
                self.dataloader = DataDistributor.make_dataloader(self.dataset, shuffle=True,
                                                                  batch_size=self.env_args.batch_size)
                self.backdoor_args.k = -1

            else:
                self._setup_backdoor()
        else:
            self.dataloader = DataDistributor.make_dataloader(self.dataset, shuffle=True,
                                                              batch_size=self.env_args.batch_size)

    def _local_eval(self):
        self.model.eval()
        # self.dataset.set_train(False)
        # self.train_accuracy = test_model(self.model, self.dataloader, self.device)
        # self.dataset.set_train(True)
        self.backdoor_asr = test_model(self.model, self.backdoor_loader_eval, self.device)
        self.clean_test_accuracy = test_model(self.model, self.clean_loader_eval, self.device)
        # print(f"Avg Training Loss: {self.loss:.3f} | Clean Accuracy: {self.clean_test_accuracy} | Backdoor ASR: {self.backdoor_asr} | Local Train Accuracy: {self.train_accuracy}")
        print(f"Avg Training Loss: {self.loss:.3f} | Clean Accuracy: {self.clean_test_accuracy} | Backdoor ASR: {self.backdoor_asr}")
    