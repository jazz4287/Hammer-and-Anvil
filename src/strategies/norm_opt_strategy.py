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
from torch.optim.lr_scheduler import CosineAnnealingLR

class NormOptStrategy(Strategy):
    def __init__(
        self,
        strategy_args: StrategyArgs,
        client_args: ClientArgs,
        aggregator_args: AggregatorArgs,
        backdoor_args: BackdoorArgs,
        env_args: EnvArgs = None,
    ):
        super(NormOptStrategy, self).__init__(
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
        best_backdoor = None
        best_clean = None
        best_combined = None
        # fetch model to train and put it on the correct device
        aggregator = self.aggregator
        aggregator.model.to(device)

        for i in range(len(self.clients)):
            client = self.clients[i]
            client.model.to(device)

            if i+1 < self.client_args.num_clients - self.client_args.num_malicious_clients:
                client.optimizer = torch.optim.SGD(client.model.parameters(), lr=self.client_args.benign_lr, momentum=0.9)
            else:
                client.optimizer = torch.optim.SGD(client.model.parameters(), lr=self.client_args.malicious_lr, momentum=0.9)
            
            client.scheduler = CosineAnnealingLR(client.optimizer, T_max=self.strategy_args.num_epochs)

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
                
                client.scheduler.step()

            # scale mal updates to max norm bound
            for mal in self.malicious_models:
                norm = NormAggregator._get_norm(aggregator_state_dict, mal)
                mal_coef = self.aggregator_args.norm_bound / norm
                for k in mal:
                    if not "num_batches_tracked" in k:
                        mal[k] = aggregator_state_dict[k] + mal_coef * (mal[k] - aggregator_state_dict[k])
                print(f"mal opt: norm before scale: {norm:.2f}, coef: {mal_coef}, norm after scale: {NormAggregator._get_norm(aggregator_state_dict, mal):.2f}")

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
        return "norm_opt_strategy"
