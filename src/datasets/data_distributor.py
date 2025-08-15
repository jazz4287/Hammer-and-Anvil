from global_settings import CACHE_DIR
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.datasets.dataset_factory import DatasetFactory
import numpy as np
from torch.utils.data import DataLoader
import os
from collections import defaultdict
import random


class DataDistributor:
    def __init__(self, client_args: ClientArgs, env_args: EnvArgs, aggregator_args: AggregatorArgs):
        self.client_args = client_args
        self.env_args = env_args
        self.aggregator_args = aggregator_args
        self.train_dataset = DatasetFactory.from_env_args(env_args, train=True)
        self.eval_dataset = DatasetFactory.from_env_args(env_args, train=False)
        self.aggregator_set = None
        # for now we assume each client has access to the same amount of data

        if aggregator_args.fine_tune > 0:
            self._total_train_dataset = self.train_dataset.copy()
            fine_tune_id_file_path = os.path.join(CACHE_DIR, env_args.dataset, f"fine_tune_cache_{aggregator_args.fine_tune}.npy")
            if os.path.exists(fine_tune_id_file_path):
                print(f"Loading cached fine-tune dataset ids from {fine_tune_id_file_path}")
                fine_tune_indices = np.load(fine_tune_id_file_path).tolist()
                self.aggregator_set = self._total_train_dataset.subset(fine_tune_indices)
            else:
                self.aggregator_set = self._total_train_dataset.random_subset(aggregator_args.fine_tune)
                fine_tune_indices = self.aggregator_set.idx
                os.makedirs(os.path.join(CACHE_DIR, env_args.dataset), exist_ok=True)
                np.save(fine_tune_id_file_path, np.array(fine_tune_indices))
                print(f"Saved cached fine-tune dataset ids to {fine_tune_id_file_path}")
            self.train_dataset = self.train_dataset.subset(list(set(range(len(self.train_dataset)))-set(fine_tune_indices)))

        indices = np.random.permutation(np.arange(len(self.train_dataset)))
        if self.env_args.use_dirichlet:
            print(f"Using dirichlet distribution")
            # non-IID data distribution of data between clients
            class_to_idx = self.train_dataset.get_class_to_idx()
            dataset_classes = list(class_to_idx.keys())
            no_classes = len(dataset_classes)
            alpha = self.env_args.dirichlet_alpha
            per_participant_indices = defaultdict(list)
            no_participants = self.client_args.num_clients
            for n in range(no_classes):
                class_size = len(class_to_idx[dataset_classes[n]])
                random.shuffle(class_to_idx[dataset_classes[n]])
                sampled_probabilities = class_size * np.random.dirichlet(
                    np.array(no_participants * [alpha]))
                for user in range(no_participants):
                    no_imgs = int(round(sampled_probabilities[user]))
                    sampled_list =  class_to_idx[dataset_classes[n]][:min(len(class_to_idx[dataset_classes[n]]), no_imgs)]
                    per_participant_indices[user].extend(sampled_list)
                    class_to_idx[dataset_classes[n]] = class_to_idx[dataset_classes[n]][min(len(class_to_idx[dataset_classes[n]]), no_imgs):]

            # now that we've assigned the image indices for each client, we can create the dataset chunks
            print([len(per_participant_indices[user]) for user in range(no_participants)])

            self.dataset_chunks = [
                self.train_dataset.subset(per_participant_indices[user])
                for user in range(no_participants)
            ]

        else:
            self.dataset_chunks = [
                self.train_dataset.subset(sub_indices)
                for sub_indices in np.split(indices, self.client_args.num_clients)
            ]


    def get_classes(self):
        return self.train_dataset.classes

    # def get_dataloaders(self, shuffle: bool = True, *args, **kwargs):
    #     return [
    #         self.get_dataloaders(
    #             dataset_chunk,
    #             shuffle=shuffle,
    #             batch_size=self.env_args.batch_size,
    #             *args,
    #             **kwargs,
    #         )
    #         for dataset_chunk in self.dataset_chunks
    #     ]

    @staticmethod
    def make_dataloader(dataset, shuffle: bool = True,  batch_size: int = 64, *args, **kwargs):
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=16,
            *args,
            **kwargs,
        )

    def get_datasets(self):
        return self.dataset_chunks

    def get_eval_dataloader(self, shuffle: bool = False, *args, **kwargs):
        return DataLoader(
            self.eval_dataset,
            shuffle=shuffle,
            batch_size=self.env_args.eval_batch_size,
            num_workers=16,
            *args,
            **kwargs,
        )

    def get_eval_dataset(self):
        return self.eval_dataset

    def get_fine_tune_set(self):
        return self.aggregator_set