from src.clients.client import Client
from src.clients.malicious_client import MaliciousClient
from src.clients.neurotoxin_client import NeurotoxinClient
from src.clients.static import StaticBackdoorClient
from src.arguments.client_args import ClientArgs
from src.arguments.env_args import EnvArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.clients.dba_client import DBAClient

from src.utils.logging import Logger


class ClientFactory:

    clients = {"benign": Client, "malicious": MaliciousClient}

    requires_backdoor_args = ["malicious"]
    attacks = {"static": StaticBackdoorClient,
               "badnets": StaticBackdoorClient, # backward compatibility}
               "dba": DBAClient,
               "neurotoxin": NeurotoxinClient
               }

    @classmethod
    def from_client_args(
        cls,
        client_args: ClientArgs,
        client_type: str = "benign",
        env_args: EnvArgs = None,
        backdoor_args: BackdoorArgs = None,
        logger: Logger = None,
    ):
        client = cls.clients.get(client_type, None)
        if client is None:
            raise ValueError(client_type)
        kwargs = {"client_args": client_args, "env_args": env_args, "logger": logger}
        if client_type in cls.requires_backdoor_args:
            kwargs["backdoor_args"] = backdoor_args
            client = cls.attacks.get(backdoor_args.attack_type, None)
            if client is None:
                raise ValueError(backdoor_args.attack_type)
        return client(**kwargs)
