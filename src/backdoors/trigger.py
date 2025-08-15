from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from abc import abstractmethod, ABC


class Trigger(ABC):
    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        self.backdoor_args = backdoor_args
        self.env_args = env_args
        self.triggers = None

    @abstractmethod
    def generate(self, num_classes) -> dict[int: int]:
        """
        Generates the trigger look-up table where for a given class idx (between 0 and num_classes) used as key it
        generates a new class idx as value.
        :return: dict{orig_class_idx: new_class_idx} ({int: int})
        """
        raise NotImplementedError
