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

class FMNStrategy(Strategy):
    def __init__(
        self,
        strategy_args: StrategyArgs,
        client_args: ClientArgs,
        aggregator_args: AggregatorArgs,
        backdoor_args: BackdoorArgs,
        env_args: EnvArgs = None,
    ):
        super(FMNStrategy, self).__init__(
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

        lr_base = self.client_args.benign_lr
        lr_max1 = min(self.client_args.benign_lr*333, 3.)
        # lr_max1 = self.client_args.benign_lr*333
        lr_max2 = min(self.client_args.benign_lr*10, 1.)
        # lr_max2 = self.client_args.benign_lr*10
        break_point = self.strategy_args.fmn_break_point
        frequency = self.strategy_args.fmn_frequency
        interleave_clean_training = self.strategy_args.fmn_interleave_clean_training
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

        for epoch in range(self.strategy_args.num_epochs):
            print(f"\nEpoch {epoch}: Starting training.")
            fmn_epoch = epoch if not interleave_clean_training else epoch // 2
            aggregator_state_dict = aggregator.model.state_dict()
            lr_peak = lr_max1 if fmn_epoch < break_point else lr_max2
            phase = fmn_epoch % frequency
            current_lr = lr_base + (lr_peak - lr_base) * (1 - abs(2 * phase / frequency - 1))

            for malicious_client in self.malicious_clients:
                for param_group in malicious_client.optimizer.param_groups:
                    param_group['lr'] = current_lr
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
                    print(f"malicious, no_backdoor={(interleave_clean_training and epoch % 2 == 0)}, lr={current_lr}")
                    client_model_state = client.train_one_epoch(aggregator_state_dict, self.benign_models,
                                                                no_backdoor=(interleave_clean_training and epoch % 2 == 0))
                    self.malicious_models.append(client_model_state)
            if self.strategy_args.fmn_grad_clipping > 0:
                # we also clip the gradient norm
                for mal in self.malicious_models:
                    norm = NormAggregator._get_norm(aggregator_state_dict, mal)
                    if norm > self.strategy_args.fmn_grad_clipping:
                        mal_coef = self.strategy_args.fmn_grad_clipping / norm
                        for k in mal:
                            if not "num_batches_tracked" in k:
                                mal[k] = aggregator_state_dict[k] + mal_coef * (mal[k] - aggregator_state_dict[k])

            for b in self.benign_models:
                print(f"benign norm: {NormAggregator._get_norm(aggregator_state_dict, b)}")
            for m in self.malicious_models:
                print(f"mal norm: {NormAggregator._get_norm(aggregator_state_dict, m)}")
            
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

            # logger.update_epoch(epoch + 1, -1, clean_test_accuracy, backdoor_asr, ga_avg_fitness)
            logger.update_epoch(epoch + 1, -1, clean_test_accuracy, backdoor_asr)
            logger.save_progress()

        logger.set_done()  # set progress to "finished"
        self.save_best(best_clean, best_backdoor, best_combined)

    def __str__(self):
        return "fed_avg"
