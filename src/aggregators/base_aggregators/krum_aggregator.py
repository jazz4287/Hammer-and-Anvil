import copy
import torch
from src.utils.models import get_model_params, set_model_params
import tqdm

from src.aggregators.aggregator import Aggregator
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.client_args import ClientArgs
from src.arguments.env_args import EnvArgs
from src.aggregators.base_aggregators.norm_aggregator import NormAggregator

# PAPERS:
# https://arxiv.org/pdf/1703.02757


class KrumAggregator(Aggregator):
    def __init__(
        self,
        aggregator_args: AggregatorArgs,
        client_args: ClientArgs,
        env_args: EnvArgs = None,
    ):
        super(KrumAggregator, self).__init__(aggregator_args, client_args, env_args)

        print("Number of clients: ", client_args.num_clients)

    def _krum_score(self, model_param_list, cur):
        # distance(cur, i) for all i != cur
        dists = []
        for i in range(len(model_param_list)):
            if i == cur:
                continue
            # dist = 0
            # for k, m, c in zip(model_param_list[0].keys(), model_param_list[i].values(), model_param_list[cur].values()):
            #     if "num_batches_tracked" in k:
            #         continue
            #     elif m.shape == torch.Size([]):
            #         # param is scalar
            #         dist += torch.abs(m - c).item()
            #     else:
            #         dist += torch.norm(m - c)
            dist = NormAggregator._get_norm(model_param_list[i], model_param_list[cur])
            dists.append(dist)
        dists = torch.tensor(dists, device=self.device)
        dists, _ = torch.sort(dists)
        # sum over n-f-2 closest distance(cur, i), for all i != cur
        return torch.sum(dists[:self.client_args.num_clients - self.client_args.num_malicious_clients - 2])
        
        # roll back
        # return torch.sum(dists[:self.client_args.num_clients - self.client_args.num_malicious_clients - 1])
    
    def _param_krum_top(self, model_param_list, m, bar):
        # compute all scores at once, then take the m lowest
        scores = []
        for i in range(len(model_param_list)):
            score = self._krum_score(model_param_list, i)
            scores.append(score)
            bar.update(1)
        scores = torch.tensor(scores)
        # print(scores)
        sorted_client_idx = torch.argsort(scores)
        print(f"krum top {m} is {sorted_client_idx[:m]}")
        new_params = self._avg_param(model_param_list, sorted_client_idx[:m])
        return new_params
    
    def _param_krum_iterative(self, model_param_list, m, bar):
        # re-compute krum score with m-1 nodes every iteration
        selected_clients = []
        map_model_param_list = [(i, model_param) for i, model_param in enumerate(model_param_list)]
        while m > 0:
            scores = []
            crop_model_param_list = [model_param for _, model_param in map_model_param_list]
            for i in range(len(crop_model_param_list)):
                score = self._krum_score(crop_model_param_list, i)
                # scores.append((score, i))
                scores.append(score)
            # scores = torch.tensor(scores, device=self.device)
            # scores, _ = torch.sort(scores, dim=0)
            scores = torch.tensor(scores)
            sorted_client_idx = torch.argsort(scores)
            top_client = sorted_client_idx[0]
            selected_clients.append(map_model_param_list[top_client][0])
            map_model_param_list.pop(top_client)
            bar.update(1)
            m -= 1
        new_params = self._avg_param(model_param_list, selected_clients)
        return new_params
    
    def get_krum_order(self, model_param_list):
        scores = []
        for i in range(len(model_param_list)):
            score = self._krum_score(model_param_list, i)
            scores.append(score)
        scores = torch.tensor(scores)
        scores_order = torch.argsort(scores)
        return scores_order
    
    def get_neighbor_order(self, model_param_list, cur):
        # argsort(dist(cur, i)) for all i != cur
        dists = []
        for i in range(len(model_param_list)):
            if i == cur:
                continue
            dist = 0
            for k, m, c in zip(model_param_list[0].keys(), model_param_list[i].values(), model_param_list[cur].values()):
                if "num_batches_tracked" in k:
                    continue
                elif m.shape == torch.Size([]):
                    # param is scalar
                    dist += torch.abs(m - c).item()
                else:
                    dist += torch.norm(m - c)
            dists.append(dist)
        dists = torch.tensor(dists, device=self.device)
        dists_order = torch.argsort(dists)
        return dists_order
    
    # Krum Aggregator
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
            total=self.aggregator_args.m_krum,
            desc="Aggregating",
            position=0,
            leave=True,
            disable=mock
        )

        if self.aggregator_args.krum_type == "top":
            new_params = self._param_krum_top(model_param_list, self.aggregator_args.m_krum, bar)
        elif self.aggregator_args.krum_type == "iterative":
            new_params = self._param_krum_iterative(model_param_list, self.aggregator_args.m_krum, bar)
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
