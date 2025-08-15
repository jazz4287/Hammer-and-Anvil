from src.backdoors.trigger import Trigger


class LabelShiftTrigger(Trigger):
    def generate(self, num_classes) -> dict[int: int]:
        self.triggers = {}
        shift_magnitude = self.backdoor_args.shift_magnitude % num_classes
        for i in range(num_classes):
            self.triggers[i] = (shift_magnitude + i) % num_classes
        return self.triggers
