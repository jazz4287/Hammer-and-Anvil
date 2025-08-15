import torch
import tqdm

from src.arguments.client_args import ClientArgs
from src.arguments.env_args import EnvArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.clients.static import StaticBackdoorClient
from src.utils.logging import Logger
from src.backdoors.base_backdoors.dba import DBABackdoor
from src.datasets.data_distributor import DataDistributor


class DBAClient(StaticBackdoorClient):

    def __init__(
            self,
            client_args: ClientArgs,
            backdoor_args: BackdoorArgs,
            env_args: EnvArgs = None,
            logger: Logger = None,
    ):
        super(DBAClient, self).__init__(client_args, backdoor_args, env_args, logger)
        # we use the client_id, the number of clients and the number of malicious clients to workout which pattern to
        # use. e.g. for n=20, m=4, mal ids are [16,17,18,19] -> patterns [0,1,2,3]
        # assert self.client_args.num_malicious_clients == 4
        self.pattern_num = None
        self.epoch_tracker = 0

    def _setup_backdoor(self):
        if self.pattern_num is None:
            # we do this here because the id is not yet set when the client is initialized
            pattern_num = (self.id - self.client_args.num_clients + self.client_args.num_malicious_clients)%4
            self.pattern_num = pattern_num
            print(self.pattern_num)
        # we first process the train dataset with this client's unique pattern
        self.dataset, _, _, self.skipped_triggers, self.poisoned, self.processed_images = self.backdoor.process_dataset(self.dataset, pattern_ids=(self.pattern_num, ))
        print(f"Client {self.id}: Skipped {self.skipped_triggers} images. Poisoned {self.poisoned} images. Processed {self.processed_images} images.")
        # general dataloader kept for training
        self.dataloader = DataDistributor.make_dataloader(self.dataset, shuffle=True, batch_size=self.env_args.batch_size)
        # test set w/ eval transforms
        # now we process the test dataset with the combined trigger
        _, self.clean_dataset, self.backdoor_dataset, _, _, _ = self.backdoor.process_dataset(self.dataset, pattern_ids=tuple(range(4)))
        print(f"Processed DBA test set with using all patterns: {tuple(range(4))}")
        self.clean_dataset.set_train(False)
        self.backdoor_dataset.set_train(False)
        self.clean_loader_eval = DataDistributor.make_dataloader(self.clean_dataset, shuffle=False, batch_size=self.env_args.eval_batch_size)
        self.backdoor_loader_eval = DataDistributor.make_dataloader(self.backdoor_dataset, shuffle=False, batch_size=self.env_args.eval_batch_size)


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

        new_model_state_dict = super().train_one_epoch(model_state_dict, *args, **kwargs)
        # start_epoch = 50
        # base_scaler = 1
        # max_scaler = 3
        # t_max = 80
        # if self.epoch_tracker < start_epoch:
        #     scaler = base_scaler
        # else:
        #     scaler =  min(float(max_scaler), (base_scaler*(t_max - self.epoch_tracker+start_epoch) + max_scaler*(self.epoch_tracker-start_epoch))/t_max)  # we are trying a progressive scaler as any fixed value is too high at the beginning and too low at the end
        # print(f"Client {self.id}: scaler: {scaler}")
        # self.epoch_tracker += 1
        if kwargs.get("scale", False): #and self.client_args.dba_single_shot:
            # only scale for single shot
            scaler = self.client_args.dba_param_scaler
            print(f"using scaler: {scaler}")
        else:
            scaler = 1


        # scaler = 3  # 10, 100 is too high
        # now we want to scale the change in weights, the paper scales by x100
        if scaler != 1:
            for k in new_model_state_dict.keys():
                if "num_batches_tracked" in k:
                    continue
                    # new_model_state_dict[k] = new_model_state_dict[k] * (self.client_args.malicious_epoch / self.client_args.benign_epoch)
                else:
                    new_param = model_state_dict[k] + scaler * (new_model_state_dict[k] - model_state_dict[k])
                    new_model_state_dict[k] = new_param

        return new_model_state_dict