from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.datasets.dataset import Dataset
import torch
from src.backdoors.trigger_factory import TriggerFactory
from torch.utils.data import TensorDataset
from abc import ABC
from copy import deepcopy
from abc import abstractmethod


class Backdoor(ABC):
    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        self.backdoor_args = backdoor_args
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.trigger = TriggerFactory.from_backdoor_args(backdoor_args)
        self.triggers = None

    def copy(self):
        """Return a copy of this dataset instance."""
        return deepcopy(self)

    def generate_triggers(self, num_classes):
        self.triggers = self.trigger.generate(num_classes)

    def process_dataset(self, dataset: Dataset, *args, **kwargs):
        dataset_copy = dataset.copy()  # so we don't modify the original dataset in case we need it again
        if self.triggers is None:
            self.generate_triggers(dataset_copy.num_classes())
            print(f"Triggers: {self.triggers}")
        data, labels = dataset_copy.get_indexed_data_and_targets()
        new_data, new_labels, total_poisoned = self.process_inputs(data, labels, *args, **kwargs)
        dataset_copy.set_dataset(TensorDataset(new_data, new_labels))
        # return dataset_copy, new_data.size(0)-total_poisoned, total_poisoned, new_data.size(0)
    
        orig_data, orig_labels = dataset.get_indexed_data_and_targets()
        clean_data, clean_labels = [], []
        backdoor_data, backdoor_labels = [], []
        for i in range(new_data.size(0)):
            if torch.equal(new_data[i], orig_data[i]) and new_labels[i] == orig_labels[i]:
                clean_data.append(orig_data[i])
                clean_labels.append(orig_labels[i])
            elif (not torch.equal(new_data[i], orig_data[i])) and new_labels[i] != orig_labels[i]:
                backdoor_data.append(new_data[i])
                backdoor_labels.append(new_labels[i])
            else:
                if hasattr(self, "patterns"):
                    # we are using the DBA setting and the trigger is already present in the image by accident
                    # patterns = []
                    # for pattern in self.patterns.values():
                    #     patterns.extend(pattern)
                    # count = 0
                    # for x_idx, y_idx in patterns:
                    #     count += orig_data[i, :, x_idx, y_idx] == 1
                    # # print(count, len(patterns))
                    # for count_val in count:
                    #     if count_val != len(patterns):
                    #         # the trigger is not already present and something went very wrong
                    #         raise ValueError(i, new_labels[i], orig_labels[i])
                    backdoor_data.append(new_data[i])
                    backdoor_labels.append(new_labels[i])
                else:
                    raise ValueError(i, new_labels[i], orig_labels[i])
        
        clean_data, clean_labels = torch.stack(clean_data, dim=0), torch.LongTensor(clean_labels)
        clean_dataset = dataset.copy()    
        clean_dataset.set_dataset(TensorDataset(clean_data, clean_labels))

        backdoor_data, backdoor_labels = torch.stack(backdoor_data, dim=0), torch.LongTensor(backdoor_labels)
        backdoor_dataset = dataset.copy()
        backdoor_dataset.set_dataset(TensorDataset(backdoor_data, backdoor_labels))

        # need to do set_train for returned dataset
        return dataset_copy, clean_dataset, backdoor_dataset, new_data.size(0)-total_poisoned, total_poisoned, new_data.size(0)

    @abstractmethod
    def process_inputs(self, images, labels, *args, **kwargs) -> (torch.Tensor, torch.Tensor, int):
        raise NotImplementedError
