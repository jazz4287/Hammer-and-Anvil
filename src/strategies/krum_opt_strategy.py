import pickle

import torch
import random

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
from src.aggregators.base_aggregators.krum_aggregator import KrumAggregator

class KrumOptStrategy(Strategy):
    def __init__(
        self,
        strategy_args: StrategyArgs,
        client_args: ClientArgs,
        aggregator_args: AggregatorArgs,
        backdoor_args: BackdoorArgs,
        env_args: EnvArgs = None,
    ):
        super(KrumOptStrategy, self).__init__(
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

        best_backdoor = None
        best_clean = None
        best_combined = None

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
        picked_benign = []

        assert(isinstance(aggregator, KrumAggregator))

        for epoch in range(self.strategy_args.num_epochs):
            print(f"\nEpoch {epoch}: Starting training.")
            aggregator_state_dict = aggregator.model.state_dict()

            # runs from client
            client_model_states = []
            self.benign_models = []
            self.malicious_models = []

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
                    self.benign_models.append(client_model_state)
                else:
                    # only possible after all selected benign clients are visited
                    print("malicious")
                    client_model_state = client.train_one_epoch(aggregator_state_dict, self.benign_models)
                    self.malicious_models.append(client_model_state)

            # just for logging purpose, will always overwrite ALL mal update with mal_center
            krum_order = aggregator.get_krum_order(self.benign_models + self.malicious_models)
            print(f"no dup, krum picked client {krum_order[0]} is benign? {krum_order[0] < len(self.benign_models)}")
            # overwrite with mal_center
            mal_center = aggregator._avg_param(self.malicious_models, range(len(self.malicious_models)))
            self.malicious_models = [mal_center for i in range(len(self.malicious_models))]
            new_update = self.benign_models + self.malicious_models
            krum_order = aggregator.get_krum_order(new_update)
            print(f"dup mal no opt, krum picked client {krum_order[0]} is benign? {krum_order[0] < len(self.benign_models)}")
            need_opt = krum_order[0] < len(self.benign_models)
            if krum_order[0] < len(self.benign_models):
                print("require opt")
                # all mal update the same (mal_center), just use the last one
                mal_nbr_order = aggregator.get_neighbor_order(new_update, len(new_update)-1)
                # should have (n-f-2)-(f-1) = n-2f-1 benign neighbours in the krum range
                closest_ben = []
                for i in range(self.client_args.num_clients - self.client_args.num_malicious_clients - 2):
                    model_idx = mal_nbr_order[i]
                    if model_idx < len(self.benign_models):
                        closest_ben.append(new_update[model_idx])
                # now have ben_center, mal_center, and current best krum score (r)
                ben_center = aggregator._avg_param(closest_ben, range(len(closest_ben)))
                radius = aggregator._krum_score(new_update, krum_order[0]).item()
                print(f"min krum score (orig radius) before mod: {radius:.2f}")

                if self.strategy_args.krum_attack == "bisect_opt_alpha":
                    left = 0.
                    right =  1.
                    center = (right+left)/2
                    max_depth = self.strategy_args.bisect_opt_alpha_depth
                    cur_depth = 0
                    last_successful = None
                    while cur_depth < max_depth:
                        # we try the center
                        opt_mal = {}
                        for k in mal_center.keys():
                            if "num_batches_tracked" in k:
                                opt_mal[k] = mal_center[k]
                            else:
                                opt_mal[k] = center * mal_center[k] + (1 - center) * ben_center[k]
                        malicious_models = [opt_mal for _ in range(len(self.malicious_models))]
                        krum_order = aggregator.get_krum_order(self.benign_models + malicious_models)[0].item()
                        print(f"depth={cur_depth}, left={left:.2e}, center={center:.2e}, right={right:.2e}, picked={krum_order}, mal={krum_order >= len(self.benign_models)}")
                        if krum_order >= len(self.benign_models):
                            # means that the id of the model is malicious
                            # if succeed we go right
                            if last_successful is None or center > last_successful:
                                last_successful = center
                            left = center
                            center = (right+left)/2
                        else:
                            right = center
                            center = (right+left)/2
                        cur_depth += 1
                    if last_successful is None:
                        print(f"Bisection was not successful. Setting alpha=0")
                        alpha = 0
                    else:
                        print(f"Bisection was successful: alpha={last_successful}")
                        alpha = last_successful
                    # raise(NotImplementedError)
                elif self.strategy_args.krum_attack == "fixed_alpha":
                    alpha = self.strategy_args.krum_fixed_alphas
                    mal_ben_center_dist = None
                elif self.strategy_args.krum_attack == "dynamic_alpha":
                    radius = radius / (len(closest_ben) * self.strategy_args.krum_dynamic_radius_scaling)
                    print(f"radius after scaling: {radius:.2f}")
                    eps = 10**-5
                    mal_ben_center_dist = NormAggregator._get_norm(mal_center, ben_center)
                    alpha = (radius - eps) / mal_ben_center_dist
                else:
                    mal_ben_center_dist = None
                    
                print(f"{alpha:.2f} * mal + {1-alpha:.2f} * ben_center")
                opt_mal = {}
                for k in mal_center.keys():
                    if "num_batches_tracked" in k:
                        opt_mal[k] = mal_center[k]
                    else:
                        opt_mal[k] = alpha * mal_center[k] + (1-alpha) * ben_center[k]
                self.malicious_models = [opt_mal for i in range(len(self.malicious_models))]
            else:
                radius = None
                alpha = None
                mal_ben_center_dist = None
            krum_order = aggregator.get_krum_order(self.benign_models + self.malicious_models)
            best_radius = aggregator._krum_score(self.benign_models + self.malicious_models, krum_order[0]).item()
            mal_radii = [aggregator._krum_score(self.benign_models + self.malicious_models, i).item() for i in range(len(self.benign_models), len(self.benign_models + self.malicious_models))]
            print(f"final result, krum picked client {krum_order[0]} is benign? {krum_order[0] < len(self.benign_models)}")
            print(f"best krum {best_radius:.2f}",  " | ".join(f"{mal_radius:.2f}" for mal_radius in mal_radii))
            if self.env_args.save_picked_benign:
                picked_benign.append((krum_order[0] < len(self.benign_models), radius, alpha, mal_ben_center_dist, need_opt))  # (picked_benign, radius, alpha, mal_ben_center_dist, picked_benign_before_opt)
            # Step 4: aggregate the parameters using the aggregator
            print(f"Epoch {epoch}: Training complete, aggregating models.")
            client_model_states = self.benign_models + self.malicious_models
            aggregator.aggregate(client_model_states)

            # TODO: move evaluation to aggregator (specific aggregators might have custom metrics they want to monitor)
            # re-poinson if needed, don't need to refresh eval backdoor data yet
            
            clean_test_accuracy = test_model(aggregator.model, aggregator.get_eval_dataloader(), device).item()
            backdoor_asr = test_model(aggregator.model, self.backdoor_eval_dataloader, device).item()
            if best_clean is None:
                best_clean = (clean_test_accuracy, backdoor_asr)
                best_backdoor = (clean_test_accuracy, backdoor_asr)
                best_combined = (clean_test_accuracy, backdoor_asr)

            if best_clean[0] < clean_test_accuracy:
                best_clean = (clean_test_accuracy, backdoor_asr)

            if best_backdoor[1] < backdoor_asr:
                best_backdoor = (clean_test_accuracy, backdoor_asr)

            if sum(best_combined) < clean_test_accuracy + backdoor_asr:
                best_combined = (clean_test_accuracy, backdoor_asr)

            print(f"Global Clean Accuracy: {clean_test_accuracy} | Backdoor ASR: {backdoor_asr} | best combined (Clean, ASR): {best_combined}")

            # save model
            # save_path = path.join(save_folder, f"epoch_{epoch}.pt")
            # torch.save(aggregator.model.state_dict(), save_path)
            # print(f"Saved model to {save_path}.")

            logger.update_epoch(epoch + 1, -1, clean_test_accuracy, backdoor_asr)
            logger.save_progress()

        logger.set_done()  # set progress to "finished"
        self.save_best(best_clean, best_backdoor, best_combined)
        if self.env_args.save_picked_benign:
            print(picked_benign)
            with open("./krum_picked_benign.pkl", "wb") as f:
                pickle.dump(picked_benign, f)

    def __str__(self):
        return "krum_opt_strategy"
