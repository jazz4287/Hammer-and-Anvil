from src.backdoors.trigger import Trigger
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
import random


class SingleLabelTrigger(Trigger):
    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)
        self.target_class = None

    def generate(self, num_classes) -> dict[int: int]:
        self.triggers = {}
        self.target_class = self.backdoor_args.target_class if (
                self.backdoor_args.target_class != -1) else random.randint(0, num_classes)
        if self.backdoor_args.target_class == -1:
            print(f"Backdoor target class is not defined, setting it to {self.target_class}")
        for i in range(num_classes):
            self.triggers[i] = self.target_class
        return self.triggers

