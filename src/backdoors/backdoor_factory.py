from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoors.backdoor import Backdoor
from src.backdoors.base_backdoors.single_pixel import SinglePixelBackdoor
from src.backdoors.base_backdoors.backdoorbox_backdoor import BackdoorBox
from src.backdoors.base_backdoors.dba import DBABackdoor
class BackdoorFactory:

    backdoors = {
        "single_pixel": SinglePixelBackdoor,
        "blended": BackdoorBox,
        "dba": DBABackdoor
    }

    cache = {}

    @classmethod
    def from_backdoor_args(cls, backdoor_args: BackdoorArgs, env_args: EnvArgs, no_cache: bool = False) -> Backdoor:
        backdoor = cls.backdoors.get(backdoor_args.backdoor_name, None)
        if backdoor is None:
            raise ValueError(backdoor_args.backdoor_name)
        if no_cache:
            return backdoor(backdoor_args, env_args)
        else:
            cached_backdoor = cls.cache.get(backdoor_args.backdoor_name, None)
            if cached_backdoor is None:
                new_backdoor = backdoor(backdoor_args, env_args)
                cls.cache[backdoor_args.backdoor_name] = new_backdoor
                return new_backdoor
            else:
                return cached_backdoor.copy()

