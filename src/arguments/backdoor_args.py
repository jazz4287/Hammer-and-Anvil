from dataclasses import dataclass, field


@dataclass
class TriggerArgs:

    trigger_type: str = field(default="clean_label", metadata={"help": "which label change should be applied",
                                                               "choices": ["clean_label", "single_label",
                                                                           "label_shift"]})
    # single_label args
    target_class: int = field(default=-1,
                              metadata={"help": "target class to misclassify to with the backdoor"})

    # label_shift args
    shift_magnitude: int = field(default=0,
                                 metadata={"help": "by how many indices should the class values be shifted"})

@dataclass
class BackdoorArgs(TriggerArgs):

    CONFIG_KEY = "backdoor_args"
    """ This class contains all parameters for the backdoor attacks. """

    acceptable_triggers = {"single_pixel": ["single_label", "label_shift"]}

    backdoor_name: str = field(
        default="",
        metadata={
            "help": "name of the backdoor algorithm to use",
            "choices": [
                "single_pixel",
                "blended",
                "dba"
            ],
        },
    )

    k: float = field(
        default=1,
        metadata={
            "help": "strength of the backdoor attack, a percentage",
        },
    )

    refresh_backdoor: bool = field(default=False, metadata={"help": "whether the backdoor should be refreshed"
                                                                    "every epoch of training"})

    attack_type: str = field(
        default="",
        metadata={
            "help": "attack type of malicious clients",
            "choices": [
                "badnets",
                "model_replacement",
                "blended",
                "dba"
            ]
        }
    )


    def __post_init__(self):
        if self.backdoor_name in self.acceptable_triggers.keys():
            if self.trigger_type not in self.acceptable_triggers[self.backdoor_name]:
                raise ValueError(f"Invalid trigger: {self.trigger_type} for backdoor: {self.backdoor_name}\n"
                                 f"Acceptable values are: {self.acceptable_triggers[self.backdoor_name]}")
