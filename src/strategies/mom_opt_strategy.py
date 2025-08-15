import torch
import random
from tqdm import tqdm
from src.strategies.strategy import Strategy
from src.arguments.strategy_args import StrategyArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.utils.models import get_new_save_folder, test_model
from src.utils.logging import create_logger
from src.utils.gpu import DEVICE
from src.aggregators.base_aggregators.norm_aggregator import NormAggregator
import numpy as np
import copy


class MoMOptStrategy(Strategy):
    def __init__(
        self,
        strategy_args: StrategyArgs,
        client_args: ClientArgs,
        aggregator_args: AggregatorArgs,
        backdoor_args: BackdoorArgs,
        env_args: EnvArgs = None,
    ):
        super(MoMOptStrategy, self).__init__(
            strategy_args, client_args, aggregator_args, backdoor_args, env_args
        )

    def run(self):
        """
        Loop over the number of epochs to train for (self.strategy_args.num_epochs) and:
        - for each client, run train_one_epoch and get the updated parameters
        - aggregate the parameters using the aggregator
        - evaluate the global model's performance by calling the aggregator.evaluate function
        :return:
        """
        # Step 1: outer loop looping over the number of epochs
        save_folder = get_new_save_folder(self.env_args.save_path)
        logger = create_logger(save_folder)
        device = torch.device(DEVICE)

        # fetch model to train and put it on the correct device
        aggregator = self.aggregator
        aggregator.model.to(device)
        for client in self.clients:
            client.model.to(device)

        logger.set_epoch(0)
        logger.set_total_epochs(self.strategy_args.num_epochs)
        logger.set_config_args(
            self.strategy_args,
            self.client_args,
            self.aggregator_args,
            self.backdoor_args,
            self.env_args,
        )

        for _, client in enumerate(self.clients):
            client.set_logger(logger)

        logger.save_progress()

        for epoch in range(self.strategy_args.num_epochs):
            print(f"\nEpoch {epoch}: Starting training.")
            aggregator_state_dict = aggregator.model.state_dict()

            # runs from client
            client_model_states = []
            benign_models = []
            malicious_models = []

            # Step 2: inner loop looping over each client
            # TODO: parallize client creation/training for multi-gpu?
            subset = random.sample(range(self.client_args.num_clients), min(self.client_args.num_clients,
                                                                            max(1, self.client_args.subset_size)))
            subset.sort()
            for i in subset:
                print(f"Training client {i} for epoch {epoch}.")
                client = self.clients[i]
                client.set_epoch(epoch)
                client.set_id(i)
                # NOTE: client already has dataloader given from strategy.py
                # Step 3: train client's local model
                if i < self.client_args.num_clients - self.client_args.num_malicious_clients:
                    print("benign")
                    client_model_state = client.train_one_epoch(aggregator_state_dict)
                    benign_models.append(client_model_state)
                else:
                    # only possible after all selected benign clients are visited
                    print("malicious")
                    client_model_state = client.train_one_epoch(aggregator_state_dict, benign_models)
                    malicious_models.append(client_model_state)

            for b in benign_models:
                print(f"benign norm: {NormAggregator._get_norm(aggregator_state_dict, b)}")
            for m in malicious_models:
                print(f"mal norm: {NormAggregator._get_norm(aggregator_state_dict, m)}")

            # we first measure the distance between the original median and the mal update before opt
            mal_center = aggregator._avg_param(malicious_models, range(len(malicious_models)))
            # mock_median = self.aggregator.aggregate(benign_models + malicious_models, mock=True).state_dict()
            # print(f"Pre-OPT distance between mock median and mal: {NormAggregator._get_norm(mal_center, mock_median)}")
            # Step 4: we first compute the median and left right means of the benign updates
            # we sample it k times to get a good estimate
            k = 1000
            # delta = 0.05
            # threshold = 0.5
            threshold = 0.7
            left_mean_aggregator = None
            right_mean_aggregator = None
            median_aggregator = None
            pbar = tqdm(range(k))
            pbar.set_description("Aggregating [left, median, right] groups")
            with torch.no_grad():
                for _ in pbar:
                    left_mean, median, right_mean = aggregator._param_median_left_right(benign_models+malicious_models)
                    if left_mean_aggregator is None:
                        left_mean_aggregator = left_mean
                        right_mean_aggregator = right_mean
                        median_aggregator = median
                    else:
                        for key in left_mean_aggregator.keys():
                            left_mean_aggregator[key] += left_mean[key].clone().detach()
                            right_mean_aggregator[key] += right_mean[key].clone().detach()
                            median_aggregator[key] += median[key].clone().detach()
                for key in left_mean_aggregator.keys():
                    left_mean_aggregator[key] /= k
                    right_mean_aggregator[key] /= k
                    median_aggregator[key] /= k

            # Step 5: compute the malicious center and overwrite the malicious models with the center
            print(f"Pre-OPT distance between estimated median and mal: {NormAggregator._get_norm(mal_center, median_aggregator)}")

            new_mal_update = {}
            # Step 6: Compute the scaler to compute the new update mal = (1-scaler)median + scaler*mal_center such that mal \in [left, right]
            mal_list = []
            median_list = []
            left_list = []
            right_list = []
            for key in mal_center.keys():
                mal_list.append(mal_center[key].flatten())
                median_list.append(median_aggregator[key].flatten())
                left_list.append(left_mean_aggregator[key].flatten())
                right_list.append(right_mean_aggregator[key].flatten())
            mal_vec = torch.cat(mal_list).to(DEVICE)
            median_vec = torch.cat(median_list).to(DEVICE)
            left_vec = torch.cat(left_list).to(DEVICE)
            right_vec = torch.cat(right_list).to(DEVICE)
            assert left_vec.sum() <= median_vec.sum() <= right_vec.sum(), print(left_vec.sum(), median_vec.sum(), right_vec.sum())
            tol = 1e-6
            if not torch.all((left_vec - tol) <= median_vec) or not torch.all(median_vec <= (right_vec + tol)):
                # Print warnings for elements that violate the ordering.
                ordering_violations = \
                (~((left_vec - tol) <= median_vec) | ~(median_vec <= (right_vec + tol))).nonzero(as_tuple=True)[0]
                for i in ordering_violations.tolist():
                    print(
                        f"Warning at index {i}: (l, med, r) = ({left_vec[i].item()}, {median_vec[i].item()}, {right_vec[i].item()})")
            # Create a boolean mask where the malicious value m is within [l, r].
            within_bounds = (mal_vec >= left_vec) & (mal_vec <= right_vec)
            print(f"Proportion in LCR pre-OPT: {sum(within_bounds)/within_bounds.size(0)}")

            # Initialize scaler tensor with default value 1.
            scalers = torch.ones_like(mal_vec, device=DEVICE)
            # Now identify elements that are out-of-bound.
            # For those, if m > median, compute scaler = (r - median) / (m - median).
            # For those, if m < median, compute scaler = (median - l) / (median - m).
            cond_upper = (~within_bounds) & (mal_vec > median_vec)
            cond_lower = (~within_bounds) & (mal_vec < median_vec)
            # Compute for the upper branch. Avoid division by zero by relying on the ordering assumptions.
            scalers[cond_upper] = (right_vec[cond_upper] - median_vec[cond_upper]) / (
                        mal_vec[cond_upper] - median_vec[cond_upper])
            scalers[cond_lower] = (median_vec[cond_lower] - left_vec[cond_lower]) / (
                        median_vec[cond_lower] - mal_vec[cond_lower])
            scaler = np.quantile(scalers.cpu().numpy(), threshold)  # for some reason torch.quantile is hela slow


            print(f"Computed scaler for MoM opt with quantile {threshold}: {scaler:.4f} (min: {scalers.min():.4f}, "
                  f"max: {scalers.max():.4f})")
            for key in mal_center.keys():
                assert key in list(median_aggregator.keys())
            for key in median_aggregator.keys():
                assert key in list(mal_center.keys())

            for key in mal_center.keys():
                new_mal_update[key] = scaler*mal_center[key] + (1-scaler) *median_aggregator[key]
            # we want to see how much the new mal update preserves the backdoor
            mock_model = copy.deepcopy(aggregator.model)
            mock_model.load_state_dict(new_mal_update)
            clean_test_accuracy = test_model(mock_model, aggregator.get_eval_dataloader(), device)
            backdoor_asr = test_model(mock_model, self.backdoor_eval_dataloader, device)
            print(f"New mal update | Clean Accuracy: {clean_test_accuracy} | Backdoor ASR: {backdoor_asr}")

            malicious_models = [new_mal_update for _ in range(len(malicious_models))]
            # Step 5: aggregate the parameters using the aggregator
            print(f"Epoch {epoch}: Training complete, aggregating models.")
            client_model_states = benign_models + malicious_models
            # aggregator.aggregate(client_model_states)
            new_left_mean, new_median, new_right_mean = aggregator._param_median_left_right(client_model_states)
            aggregator.model.load_state_dict(new_median)
            print(f"Post aggregate & opt distance between expected median and post opt median: {NormAggregator._get_norm(median_aggregator, new_median)}")
            print(f"Post aggregate & opt distance between median and mal: {NormAggregator._get_norm(new_mal_update, new_median)}")
            print(f"Post aggregate & opt distance between expected median and mal: {NormAggregator._get_norm(new_mal_update, median_aggregator)}")

            new_left_list = []
            right_list = []
            new_mal_list = []
            for key in mal_center.keys():
                new_left_list.append(new_left_mean[key].flatten())
                right_list.append(new_right_mean[key].flatten())
                new_mal_list.append(new_mal_update[key].flatten())
            new_left_vec = torch.cat(left_list).to(DEVICE)
            new_right_vec = torch.cat(right_list).to(DEVICE)
            new_mal_vec = torch.cat(new_mal_list).to(DEVICE)
            new_within_bounds = (new_mal_vec >= new_left_vec) & (new_mal_vec <= new_right_vec)
            print(f"Proportion in LCR post-OPT: {sum(new_within_bounds)/new_within_bounds.size(0)}")

            # TODO: move evaluation to aggregator (specific aggregators might have custom metrics they want to monitor)
            # re-poinson if needed, don't need to refresh eval backdoor data yet

            clean_test_accuracy = test_model(aggregator.model, aggregator.get_eval_dataloader(), device)
            backdoor_asr = test_model(aggregator.model, self.backdoor_eval_dataloader, device)
            print(f"Global Clean Accuracy: {clean_test_accuracy} | Backdoor ASR: {backdoor_asr}")

            # save model
            # save_path = path.join(save_folder, f"epoch_{epoch}.pt")
            # torch.save(aggregator.model.state_dict(), save_path)
            # print(f"Saved model to {save_path}.")

            # logger.update_epoch(epoch + 1, -1, clean_test_accuracy, backdoor_asr, ga_avg_fitness)
            logger.update_epoch(epoch + 1, -1, clean_test_accuracy, backdoor_asr)
            logger.save_progress()
        logger.set_done()  # set progress to "finished"

    def __str__(self):
        return "mom_opt"
