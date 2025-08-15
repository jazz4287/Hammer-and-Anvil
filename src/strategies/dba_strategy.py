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


class DBAStrategy(Strategy):
    def __init__(
            self,
            strategy_args: StrategyArgs,
            client_args: ClientArgs,
            aggregator_args: AggregatorArgs,
            backdoor_args: BackdoorArgs,
            env_args: EnvArgs = None,
    ):
        super(DBAStrategy, self).__init__(
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
        start_poison = self.client_args.dba_poison_start_epoch
        if start_poison == -1:
            # no range poisoning
            start_poison = self.strategy_args.num_epochs + 1
        if self.client_args.dba_poison_end_epoch == -1:
            end_poison = self.strategy_args.num_epochs
        else:
            end_poison = self.client_args.dba_poison_end_epoch

        use_single_shot = False
        if self.client_args.dba_single_shot:
            client_0_epoch = self.client_args.dba_client_0_epoch
            client_1_epoch = self.client_args.dba_client_1_epoch
            client_2_epoch = self.client_args.dba_client_2_epoch
            client_3_epoch = self.client_args.dba_client_3_epoch
            use_single_shot = True

        for _, client in enumerate(self.clients):
            client.set_logger(logger)

        logger.save_progress()

        for epoch in range(self.strategy_args.num_epochs):
            is_poison = epoch >= start_poison and epoch <= end_poison
            print(f"\nEpoch {epoch}: Starting training.")
            aggregator_state_dict = aggregator.model.state_dict()

            # runs from client
            client_model_states = []
            self.benign_models = []
            self.malicious_models = []

            # Step 2: inner loop looping over each client
            # TODO: parallize client creation/training for multi-gpu?
            # we check if the epoch is a poison epoch
            if is_poison and not use_single_shot:
                print("poisoning for multiple shots")
                subset = [dba_client.id for dba_client in self.malicious_clients]
                subset.extend(random.sample(range(self.client_args.num_clients), min(self.client_args.num_clients,
                                                                                max(1, self.client_args.subset_size - len(subset)))))
            elif use_single_shot:
                subset = []
                if epoch == client_0_epoch:
                    pattern_clients = [mal_client.id for mal_client in self.malicious_clients if mal_client.pattern_num == 0]
                    subset.extend(pattern_clients)
                if epoch == client_1_epoch:
                    pattern_clients = [mal_client.id for mal_client in self.malicious_clients if mal_client.pattern_num == 1]
                    subset.extend(pattern_clients)
                if epoch == client_2_epoch:
                    pattern_clients = [mal_client.id for mal_client in self.malicious_clients if mal_client.pattern_num == 2]
                    subset.extend(pattern_clients)
                if epoch == client_3_epoch:
                    pattern_clients = [mal_client.id for mal_client in self.malicious_clients if mal_client.pattern_num == 3]
                    subset.extend(pattern_clients)
                subset.extend(random.sample(range(self.client_args.num_clients), min(self.client_args.num_clients,
                                                                                max(1, self.client_args.subset_size- len(subset)))))
            else:
                subset = random.sample(range(self.client_args.num_clients), min(self.client_args.num_clients,
                                                                                max(1, self.client_args.subset_size)))
            assert len(subset) == self.client_args.subset_size
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
                    print(f"malicious, is poison: {is_poison} - {use_single_shot}")
                    # we check if we are in a poison epoch
                    if not use_single_shot and is_poison:
                        no_backdoor = not is_poison
                        # we only scale for the first epoch
                        # if epoch == start_poison:
                        #     scale = True
                        # else:
                        #     scale = False
                        scale = True
                    elif use_single_shot and is_poison:
                        if epoch == client_0_epoch and client.pattern_num == 0:
                            no_backdoor = False
                            scale = True
                        elif epoch == client_1_epoch and client.pattern_num == 1:
                            no_backdoor = False
                            scale = True
                        elif epoch == client_2_epoch and client.pattern_num == 2:
                            no_backdoor = False
                            scale = True
                        elif epoch == client_3_epoch and client.pattern_num == 3:
                            no_backdoor = False
                            scale = True
                        else:
                            no_backdoor = True
                            scale = False
                    else:
                        no_backdoor = True
                        scale = False
                    client_model_state = client.train_one_epoch(aggregator_state_dict, self.benign_models,
                                                                no_backdoor=no_backdoor, scale=scale)
                    self.malicious_models.append(client_model_state)

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
            print(
                f"Global Clean Accuracy: {clean_test_accuracy} | Backdoor ASR: {backdoor_asr} | best combined (Clean, ASR): {best_combined}")

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
