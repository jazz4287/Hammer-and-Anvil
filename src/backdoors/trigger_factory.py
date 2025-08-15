from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoors.trigger import Trigger
from src.backdoors.base_triggers.single_label import SingleLabelTrigger
from src.backdoors.base_triggers.label_shift import LabelShiftTrigger
from src.backdoors.base_triggers.clean_label import CleanLabelTrigger


class TriggerFactory:
    triggers = {
        "clean_label": CleanLabelTrigger,
        "single_label": SingleLabelTrigger,
        "label_shift": LabelShiftTrigger
    }

    @classmethod
    def from_backdoor_args(cls, backdoor_args: BackdoorArgs) -> Trigger:
        trigger = cls.triggers.get(backdoor_args.trigger_type, None)
        if trigger is None:
            raise ValueError
        return trigger(backdoor_args)
