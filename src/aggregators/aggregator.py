from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.models.model_factory import ModelFactory
import torch

class Aggregator:
    def __init__(
        self,
        aggregator_args: AggregatorArgs,
        client_args: ClientArgs,
        env_args: EnvArgs = None,
        device=None,
    ):
        self.device = (
            device
            if device
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.aggregator_args = aggregator_args
        self.client_args = client_args
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.eval_dataloader = None
        self.eval_dataset = None
        self.fine_tune_set = None
        self.fine_tune_loader = None
        self.model = ModelFactory.from_client_args(client_args, env_args=env_args)

    def _avg_param(self, model_param_list, client_ids):
        new_params = {}
        for k in model_param_list[0].keys():
            new_params[k] = torch.zeros_like(model_param_list[0][k], device=self.device)
            for i in client_ids:
                # print(f"sum: {k}, type should be {model_param_list[0][k].dtype} from list[0], newparams[key] is {new_params[k].dtype}, list[i] is {model_param_list[i][k].dtype}")
                new_params[k] += model_param_list[i][k]
            # print(f"avg: {k}, type should be {model_param_list[0][k].dtype} from list[0], newparams[key] before avg is {new_params[k].dtype}")
            new_params[k] = (new_params[k] / len(client_ids)).to(device=self.device).to(model_param_list[0][k].dtype)
            # print(f"avg: {k}, type should be {model_param_list[0][k].dtype} from list[0], newparams[key] after avg is {new_params[k].dtype}")
        return new_params

    def set_fine_tune_set_and_dataloader(self, dataset, loader):
        self.fine_tune_set = dataset
        self.fine_tune_loader = loader

    def get_fine_tune_dataloader(self):
        return self.fine_tune_loader

    def set_dataloader(self, eval_dataloader):
        self.eval_dataloader = eval_dataloader

    def set_dataset(self, eval_dataset):
        self.eval_dataset = eval_dataset

    def set_model_state(self, model_weights):
        self.model.load_state_dict(model_weights)

    def get_eval_dataloader(self):
        return self.eval_dataloader
    
    def get_eval_dataset(self):
        return self.eval_dataset

    def aggregate(self, *args, **kwargs):
        """
        Aggregates client updates
        :return:
        """
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
