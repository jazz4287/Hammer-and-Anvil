import copy
import torch
from src.utils.models import get_model_params, set_model_params
import tqdm
import math

from src.aggregators.aggregator import Aggregator
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.client_args import ClientArgs
from src.arguments.env_args import EnvArgs


class NormAggregator(Aggregator):
    def __init__(
        self,
        aggregator_args: AggregatorArgs,
        client_args: ClientArgs,
        env_args: EnvArgs = None,
        device=None,
    ):
        super(NormAggregator, self).__init__(aggregator_args, client_args, env_args)
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

        print("Number of clients: ", client_args.num_clients)

    def _avg_param(self, model_param_list, client_ids):
        new_params = {}
        for k in model_param_list[0].keys():
            new_params[k] = torch.zeros_like(model_param_list[0][k], device=self.device)
            for i in client_ids:
                new_params[k] += model_param_list[i][k]
            new_params[k] = (new_params[k] / len(client_ids)).to(device=self.device).to(model_param_list[0][k].dtype)
        return new_params
    @staticmethod
    def _get_norm(m1, m2):
        ss = 0
        for k in m1.keys():
            if not "num_batches_tracked" in k:
                ss += torch.sum((m1[k] - m2[k])**2).item()
        return math.sqrt(ss)

    def _drop_norm(self, model_param_list, bar):
        old_model_param = self.model.state_dict()
        client_ids = []
        for i in range(len(model_param_list)):
            norm = self._get_norm(old_model_param, model_param_list[i])
            # print(norm)
            if norm < self.aggregator_args.norm_bound:
                client_ids.append(i)
            bar.update(1)
        # possible that all norms are too large, and thus dropped
        if not client_ids:
            return old_model_param
        new_params = self._avg_param(model_param_list, client_ids)
        return new_params

    def _clip_norm(self, model_param_list, bar):
        old_model_param = self.model.state_dict()
        for model_param in model_param_list:
            norm = self._get_norm(old_model_param, model_param)
            coef = min(1, self.aggregator_args.norm_bound / norm)
            for k in model_param:
                if not "num_batches_tracked" in k:
                    model_param[k] = old_model_param[k] + coef * (model_param[k] - old_model_param[k])
            print(f"<norm_agg> norm before: {norm:.2f}, coef: {coef:.2f}, norm after clipping: {self._get_norm(old_model_param, model_param):.2f}")
            bar.update(1)
        new_params = self._avg_param(model_param_list, range(len(model_param_list)))
        return new_params

    # Norm Aggregator
    def aggregate(self, model_param_list, mock=False, *args, **kwargs):
        """
        Taking in the list of client state dictionaries. It should update the global model's parameter state dictionary
        by taking the mean of the client's parameters.
        :param model_param_list: list of pytorch model's state dictionaries to aggregate.
        :param args:
        :param kwargs:
        :return:
        """

        bar = tqdm.tqdm(
            total=len(model_param_list),
            desc="Aggregating",
            position=0,
            leave=True,
            disable=mock
        )

        if self.aggregator_args.norm_type == "clip":
            new_params = self._clip_norm(model_param_list, bar)
        elif self.aggregator_args.norm_type == "drop":
            new_params = self._drop_norm(model_param_list, bar)
        else:
            raise NotImplementedError

        bar.close()
        
        if mock:
            mock_model = copy.deepcopy(self.model)
            mock_model.load_state_dict(new_params)
            return mock_model
        else:
            # update model
            self.model.load_state_dict(new_params)

    def evaluate(self):
        """Measure accuracy and loss on the evaluation dataset of the global model and print those results."""
        pass
