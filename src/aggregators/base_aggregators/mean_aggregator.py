import copy
import torch
from src.utils.models import get_model_params, set_model_params
import tqdm
import numpy as np

from src.aggregators.aggregator import Aggregator
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.client_args import ClientArgs
from src.arguments.env_args import EnvArgs

# PAPERS:
# https://arxiv.org/pdf/2202.11842.pdf
# https://proceedings.mlr.press/v195/minsker23a/minsker23a.pdf


class MeanAggregator(Aggregator):
    def __init__(
        self,
        aggregator_args: AggregatorArgs,
        client_args: ClientArgs,
        env_args: EnvArgs = None,
        device=None,
    ):
        super(MeanAggregator, self).__init__(aggregator_args, client_args, env_args)
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
        
        # N = client_args.num_clients  # sample size
        # mom_k = self.aggregator_args.mom_k  # tunable for bounds
        # m = np.floor(N / mom_k)
        # mom_l = np.log2(m)
        # min_block_size = 3
        # self.n_blocks = max(  # at LEAST one block
        #     1,  # at MOST the number of blocks that reach some minimum size
        #     min(
        #         np.floor(N / min_block_size),
        #         np.floor(mom_k * mom_l),
        #     ),
        # )
        # # NOTE: n_blocks should yield some block size of at least 3, unless there aren't enough clients,
        # # and n/(lk) where k is some tunable constant and l grows logarithmically with the number of clients
        
        self.n_blocks = self.aggregator_args.mom_blocks
        self.rmom_n = 10  # number of iterations for robust median of means
        print("Number of blocks: ", self.n_blocks)
        print("Number of clients: ", client_args.num_clients)


    def _median_of_means(self, param_list):
        params = torch.stack(param_list)
        with torch.no_grad():
            # Shuffle the indices to randomly partition the data
            indices = torch.randperm(params.size(0))
            blocks = torch.tensor_split(params[indices], self.n_blocks, dim=0)
            # Compute the mean for each block
            block_means = [torch.mean(block, dim=0) for block in blocks]
            # Compute the median of these block means and store it
            torch.use_deterministic_algorithms(False)  # does not affect correctness
            out = torch.median(torch.stack(block_means), dim=0).values
            torch.use_deterministic_algorithms(True)
            return out

    def get_block_means(self, param_list):
        """
        Splits the list of client parameter tensors into blocks,
        computes the mean of each block, and returns the list of block means.
        """
        params = torch.stack(param_list)
        with torch.no_grad():
            # Shuffle the indices to randomly partition the data into groups
            indices = torch.randperm(params.size(0))
            blocks = torch.tensor_split(params[indices], self.n_blocks, dim=0)
            block_means = [torch.mean(block, dim=0) for block in blocks]
        return block_means

    def _param_median_left_right(self, model_param_list):
        """
        For each parameter key:
          1. Compute the block (group) means from the client parameter tensors.
          2. Stack and sort these block means elementwise.
          3. Compute the median using the block means.
          4. Select the adjacent block means: one immediately left and one immediately right of the median.
             - For a single block, all values equal that block.
             - For an odd number of blocks, the left value is the group at index n//2 - 1,
               and the right value is at index n//2 + 1.
             - For an even number of blocks, the left value is the block at index n//2 - 1, and the right
               is the block at index n//2.
        Returns three dictionaries: left_means, medians, right_means.
        """
        left_values = {}
        medians = {}
        right_values = {}
        with torch.no_grad():
            for key in model_param_list[0].keys():
                # Gather and convert the client parameter values for the current parameter key.
                client_param_list = [model_params[key].float().clone().detach() for model_params in model_param_list]
                # Compute the block means.
                block_means = self.get_block_means(client_param_list)
                group_means = torch.stack(block_means)  # shape: (n_blocks, ...)
                # Sort the block means along the block dimension for each parameter element.
                sorted_vals, _ = torch.sort(group_means, dim=0)
                n = group_means.size(0)
                # Compute the median, left, and right values based on the number of blocks.
                if n == 1:
                    median = sorted_vals[0]
                    left = sorted_vals[0]
                    right = sorted_vals[0]
                elif n % 2 == 1:
                    # Odd number of blocks: take the middle block as median.
                    median = sorted_vals[n // 2]
                    # Select the immediate neighbors: one to the left and one to the right.
                    left = sorted_vals[n // 2 - 1] if n // 2 - 1 >= 0 else sorted_vals[n // 2]
                    right = sorted_vals[n // 2 + 1] if n // 2 + 1 < n else sorted_vals[n // 2]
                else:
                    # Even number of blocks: median is the average of the two middle blocks.
                    median = 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])
                    left = sorted_vals[n // 2 - 1]
                    right = sorted_vals[n // 2]

                #TODO: remove when done debugging
                # m_f = median.flatten()
                # l_f = left.flatten()
                # r_f = right.flatten()
                # for i in range(len(m_f)):
                #     if not (l_f[i] <= m_f[i] <= r_f[i]):
                #         print("Warning: ", i, (m_f[i], l_f[i], r_f[i]))
                left_values[key] = left
                medians[key] = median
                right_values[key] = right
        return left_values, medians, right_values

    # TODO:
    # def _trimmed_mean(self, seq):
    #     upperb = np.mean(seq) + np.std(seq)
    #     lowerb = np.mean(seq) - np.std(seq)
    #     seq = seq[seq <= upperb]
    #     seq = seq[seq >= lowerb]
    #     return np.mean(seq)


    def _param_median_of_means(self, model_param_list, bar, robust=False):
        print(f"MoM w/ {self.n_blocks} blocks")
        new_params = {}
        with torch.no_grad():
            for key in model_param_list[0].keys():
                client_param_list = [model_params[key].float() for model_params in model_param_list]
                robust_accumulator = torch.zeros_like(client_param_list[0]).to(self.device)
                iterations = self.rmom_n if robust else 1
                for _ in range(iterations):
                    robust_accumulator += self._median_of_means(client_param_list)
                new_params[key] = robust_accumulator if iterations == 1 else robust_accumulator / iterations
                bar.update(1)
        return new_params


    def _param_emp_mean_update(self, model_param_list, bar):
        new_params = {}
        with torch.no_grad():
            # assumes at least 1 model in the list
            for key in model_param_list[0].keys():
                accumulator = torch.zeros_like(model_param_list[0][key]).to(self.device)
                for model_params in model_param_list:
                    accumulator += model_params[key]
                mean_params = accumulator / len(model_param_list)
                new_params[key] = mean_params
                bar.update(1)
        return new_params
    

    # Median of Means Aggregator
    def aggregate(self, model_param_list, mock=False, *args, **kwargs):
        """
        Taking in the list of client state dictionaries. It should update the global model's parameter state dictionary
        by taking the mean of the client's parameters.
        :param model_param_list: list of pytorch model's state dictionaries to aggregate.
        :param args:
        :param kwargs:
        :return:
        """
        model_params = self.model.state_dict()

        # sanity check (one client changes nothing)
        if self.n_blocks == 1:
            print("WARNING: You only have one block. There will be no median of means.")

        # sanity check (there's at least two clients)
        if self.n_blocks == 2:
            print("WARNING: You have the MINIMUM number (2) of required blocks for MoM.")

        bar = tqdm.tqdm(
            total=len(list(model_params.keys())),
            desc="Aggregating",
            position=0,
            leave=True,
            disable=mock
        )

        if self.aggregator_args.mean_type == "emp":
            new_params = self._param_emp_mean_update(model_param_list, bar)
        elif self.aggregator_args.mean_type == "median_of_means":
            new_params = self._param_median_of_means(model_param_list, bar, robust=False)
        elif self.aggregator_args.mean_type == "rob_median_of_means":
            new_params = self._param_median_of_means(model_param_list, bar, robust=True)
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
