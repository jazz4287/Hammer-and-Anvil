from src.backdoors.trigger import Trigger


class CleanLabelTrigger(Trigger):
    """
    For clean label backdoor attacks, does not change the label
    """
    def generate(self, num_classes) -> dict[int: int]:
        self.triggers = {}
        for label in range(num_classes):
            self.triggers[label] = label
        return self.triggers
