import json
import os.path as path
import time

from enum import Enum
from typing import Annotated, List, Optional
from annotated_types import Ge

from pydantic import BaseModel
from pydantic.config import ConfigDict

from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.strategy_args import StrategyArgs


# TODO: implement
class History(BaseModel):
    train_loss: Annotated[float, Ge(0)]
    train_accuracy: Annotated[float, Ge(0)]
    backdoor_loss: Annotated[float, Ge(0)]
    backdoor_accuracy: Annotated[float, Ge(0)]
    metadata: dict


class ClientInfo(BaseModel):
    client_id: int  # eg. 0, 1, ... N
    epoch: int
    history: List[History]


TrainingStatus = Enum("TrainingStatus", ["unknown", "training", "finished"])


class TrainingProgress(BaseModel):
    model_config = ConfigDict(strict=True)

    env_args: EnvArgs
    client_args: ClientArgs
    aggregator_args: AggregatorArgs
    backdoor_args: BackdoorArgs
    strategy_args: StrategyArgs

    save_folder: str
    state: TrainingStatus = TrainingStatus.unknown  # eg. "training", "finished"
    current_epoch: int = 0
    total_epochs: int = -1
    # global_test_losses: List[Annotated[float, Ge(0)]]
    # global_test_accuracies: List[Annotated[float, Ge(0)]]
    global_test_losses: List[float]
    global_test_accuracies: List[float]
    backdoor_asrs: List[float]
    # ga_avg_fitness: List[float]
    clients: dict[int, ClientInfo]

    last_modified: str = "unknown"
    training_start_time: str = "unknown"


def time_now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    """
    The Logger class is responsible for saving and loading the progress of a model.

    Attributes:
        save_folder (str): The folder path where the progress log and other files will be saved.
        progress_file (str): The file path of the progress log file.
        progress (Progress): An instance of the Progress class that represents the model's progress.

    Methods:
        __init__(self, save_folder): Initializes a new instance of the Logger class.
        save_progress(self): Saves the progress of the model to a JSON file.
        load_progress(self): Loads the progress from a JSON file.
        get(self, key): Retrieves the value associated with the given key from the progress.

    """

    def __init__(self, save_folder, read_only=False, auto_load_progress=False):
        self.save_folder = save_folder
        self.progress_file = path.join(save_folder, "progress.log")
        self.progress = TrainingProgress(
            env_args=EnvArgs(),
            client_args=ClientArgs(),
            aggregator_args=AggregatorArgs(),
            backdoor_args=BackdoorArgs(),
            strategy_args=StrategyArgs(),
            save_folder=save_folder,
            global_test_losses=[],
            global_test_accuracies=[],
            backdoor_asrs=[],
            # ga_avg_fitness=[],
            clients={},
        )
        self.autoload_succeeded = True
        self.read_only = read_only
        if auto_load_progress:
            self.autoload_succeeded = self.load_progress()
        else:
            # only auto update if not loading progress
            if not self.read_only:
                print("Updating training start time.")
                self.progress.training_start_time = time_now()

    def save_progress(self, verbose=True):
        """
        Saves the progress of the model to a JSON file.

        This method serializes the model's progress into a JSON format and saves it to a file.

        Args:
            None

        Returns:
            None
        """
        if self.read_only:
            print("Read only mode, skipping save progress.")
            return
        if verbose:
            print("Saving progress...")
        self.progress.last_modified = time_now()
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.progress.model_dump_json(indent=2), f)

    def load_progress(self):
        """
        Loads the progress from a JSON file.

        Returns:
            bool: True if the progress was successfully loaded, False otherwise.
        """
        # print("Loading progress...")
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                body = json.load(f)
                self.progress = TrainingProgress.model_validate_json(body)
        except FileNotFoundError:
            return False
        # except Exception as e:
        #     print(f"Error loading progress: {e}")
        #     return False
        return True

    def set_config_args(
        self,
        strategy_args: StrategyArgs,
        client_args: ClientArgs,
        aggregator_args: AggregatorArgs,
        backdoor_args: BackdoorArgs,
        env_args: EnvArgs,
    ):
        self.progress.strategy_args = strategy_args
        self.progress.client_args = client_args
        self.progress.aggregator_args = aggregator_args
        self.progress.backdoor_args = backdoor_args
        self.progress.env_args = env_args

    def update_epoch(
        self, current_epoch: int, global_test_loss: float, global_test_accuracy: float, backdoor_asr: float, ga_avg_fitness: float=None
    ):
        self.progress.current_epoch = current_epoch
        self.progress.global_test_losses.append(global_test_loss)
        self.progress.global_test_accuracies.append(global_test_accuracy)
        self.progress.backdoor_asrs.append(backdoor_asr)
        # self.progress.ga_avg_fitness.append(ga_avg_fitness)

        self.progress.state = TrainingStatus.training

    def update_client(
        self,
        client_id: int,
        epoch: int,
        test_loss: float,
        test_accuracy: float,
        backdoor_loss: float,
        backdoor_accuracy: float,
        metadata=None,
    ):
        if client_id not in self.progress.clients:
            self.progress.clients[client_id] = ClientInfo(
                client_id=client_id,
                epoch=epoch,
                history=[],
            )

        self.progress.clients[client_id].history.append(
            History(
                train_loss=test_loss,
                train_accuracy=test_accuracy,
                backdoor_loss=backdoor_loss,
                backdoor_accuracy=backdoor_accuracy,
                metadata={},  # TODO: implement
            )
        )

        self.progress.state = TrainingStatus.training

    def set_done(self):
        self.progress.state = TrainingStatus.finished

    def set_epoch(self, current_epoch: int):
        self.progress.current_epoch = current_epoch

    def set_total_epochs(self, total_epochs: int):
        self.progress.total_epochs = total_epochs

    def log_client(self, id: int, epoch: int, history: History):
        """
        Logs the progress of a client.

        Args:
            info (ClientInfo): The information to log.

        Returns:
            None
        """
        if id not in self.progress.clients:
            self.progress.clients[id] = ClientInfo(
                client_id=id,
                epoch=epoch,
                history=[],
            )
        self.progress.clients[id].epoch = epoch
        self.progress.clients[id].history.append(history)

    def get(self, key):
        """
        Retrieves the value associated with the given key from the progress.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            Any: The value associated with the given key, or None if the key is not found.
        """
        # TODO:
        return self.progress.get(key, None)


def create_logger(save_folder):
    """
    Creates a logger object for logging purposes.

    Args:
        save_folder (str): The folder path where the log files will be saved.

    Returns:
        Logger: The logger object.

    """
    return Logger(save_folder, read_only=False)
