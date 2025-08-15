import torch
from torch.utils.data import TensorDataset
import copy
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.arguments.strategy_args import StrategyArgs
from src.arguments.env_args import EnvArgs
from src.arguments.client_args import ClientArgs
from src.arguments.aggregator_args import AggregatorArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.aggregators.aggregator_factory import AggregatorFactory
from src.clients.client_factory import ClientFactory
from src.datasets.data_distributor import DataDistributor
from src.backdoors.backdoor_factory import BackdoorFactory
from src.utils.logging import Logger
import torch.nn.functional as F
from src.utils.models import test_model, get_new_save_folder
from src.utils.gpu import DEVICE
from src.utils.logging import create_logger
from global_settings import CACHE_DIR
import json


class Strategy:
    def __init__(
        self,
        strategy_args: StrategyArgs,
        client_args: ClientArgs,
        aggregator_args: AggregatorArgs,
        backdoor_args: BackdoorArgs,
        env_args: EnvArgs = None,
        logger: Logger = None
    ):
        self.strategy_args = strategy_args
        self.client_args = client_args
        self.aggregator_args = aggregator_args
        self.backdoor_args = backdoor_args
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.logger = logger

        # initialize the aggregator
        self.aggregator = AggregatorFactory.from_aggregator_args(
            self.aggregator_args, self.client_args, env_args=self.env_args
        )
        # initialize the benign clients
        self.benign_clients = [
            ClientFactory.from_client_args(
                self.client_args,
                client_type="benign",
                env_args=self.env_args,
                logger=self.logger,
            )
            for _ in range(
                self.client_args.num_clients - self.client_args.num_malicious_clients
            )
        ]
        # initialize the malicious clients
        self.malicious_clients = [
            ClientFactory.from_client_args(
                self.client_args,
                client_type="malicious",
                env_args=self.env_args,
                backdoor_args=self.backdoor_args,
                logger=self.logger,
            )
            for _ in range(self.client_args.num_malicious_clients)
        ]
        # for convenience, we group them in a singular list when we want to iterate over all of them regardless of
        # whether they are malicious or not
        self.clients = self.benign_clients + self.malicious_clients
        for i, client in enumerate(self.clients):
            client.set_id(i)
        # setup the data distributor so we can set our client's dataloaders
        self.data_distributor = DataDistributor(
            client_args=self.client_args, env_args=self.env_args, aggregator_args=aggregator_args
        )
        # get and set the client's dataloaders
        # we also provide the clients with the dataset in case they want to perform custom preprocessing
        # this is in particular useful for malicious clients that want to perturb images
        datasets = self.data_distributor.get_datasets()
        for client, dataset in zip(self.clients, datasets):
            client.set_dataset(dataset)
        # set the aggregator's evaluation dataloader so that the evaluator can run some evaluation code on the
        # aggregated model
        self.aggregator.set_dataloader(
            self.data_distributor.get_eval_dataloader(shuffle=False)
        )
        self.aggregator.set_dataset(self.data_distributor.get_eval_dataset())

        if aggregator_args.fine_tune > 0:
            self.aggregator.set_fine_tune_set_and_dataloader(self.data_distributor.get_fine_tune_set(),
                                                         self.data_distributor.make_dataloader(
                                                             self.data_distributor.get_fine_tune_set(), shuffle=True,
                                                             batch_size=aggregator_args.fine_tune_batch_size))
        
        # set the backdoor for eval dataset
        if self.backdoor_args.attack_type:
            eval_backdoor_args = copy.deepcopy(self.backdoor_args)
            eval_backdoor_args.k = 1
            backdoor = BackdoorFactory.from_backdoor_args(eval_backdoor_args, self.env_args, no_cache=True)
            _new_dataset, _clean_dataset, backdoor_dataset, skipped_triggers, poisoned, processed_images = backdoor.process_dataset(self.aggregator.get_eval_dataset())
            print(f"Strategy: Skipped {skipped_triggers} images. Poisoned {poisoned} images. Processed {processed_images} images.")
            self.backdoor_eval_dataloader = DataDistributor.make_dataloader(backdoor_dataset, shuffle=False, batch_size=self.env_args.batch_size)

    def run(self):
        """
        Should be the function where most of the code runs. It should initialize all the missing components and
        train in a federated learning manner.
        :return:
        """
        raise NotImplementedError

    def save_best(self, best_clean, best_backdoor, best_combined):
        print(
            f"(Clean, ASR) | Best Clean: {best_clean} | Best backdoor: {best_backdoor} | best combined: {best_combined}")
        file_path = os.path.join(CACHE_DIR, "train_results.json")
        os.makedirs(CACHE_DIR, exist_ok=True)
        if not os.path.exists(file_path):
            results = {}
        else:
            with open(file_path, "r") as f:
                results = json.load(f)
        keys = [self.client_args.num_clients, self.aggregator_args.fine_tune, self.client_args.num_malicious_clients,
                self.client_args.benign_lr, self.client_args.malicious_lr]
        current_dict = results
        for key in keys:
            if current_dict.get(key, None) is None:
                current_dict[key] = {}
                current_dict = current_dict[key]
            else:
                current_dict = current_dict[key]
        current_dict["best_clean"] = best_clean
        current_dict["best_backdoor"] = best_backdoor
        current_dict["best_combined"] = best_combined

        with open(file_path, "w") as f:
            json.dump(results, f)

    def fine_tune(self, pretrained_model = None):
        if pretrained_model is not None:
            trained = torch.load(pretrained_model)
            self.aggregator.model.load_state_dict(trained)
        else:
             model = self.aggregator.model

        self.aggregator.model.to(self.aggregator.device)

        fine_tune_dataloader = self.aggregator.get_fine_tune_dataloader()
        self.aggregator.model.train()
        optimizer = self.clients[0].get_optimizer(self.aggregator.model.parameters(), self.aggregator_args.fine_tune_optimizer,
                                                  self.aggregator_args.fine_tune_lr)
        losses = 0.0
        count = 0
        num_epochs = self.aggregator_args.fine_tune_epochs
        fine_tune_size = self.aggregator_args.fine_tune
        learning_rate = self.aggregator_args.fine_tune_lr
        avg_loss = 0
        clean_test_accuracy = 0
        backdoor_asr = 0
        print(f"Fine-tuning for {self.aggregator_args.fine_tune_epochs} epochs with {self.aggregator_args.fine_tune} "
              f"samples")
        for epoch in range(self.aggregator_args.fine_tune_epochs):
            for _, (image, target) in enumerate(fine_tune_dataloader):
                # yeet features and labels to gpu
                image, target = image.to(self.aggregator.device), target.to(self.aggregator.device)

                # forward
                output = self.aggregator.model(image)
                loss = F.cross_entropy(output, target)
                losses += float(loss.item()) * image.size(0)
                count += image.size(0)
                optimizer.zero_grad()

                # back prop
                loss.backward()

                # update
                optimizer.step()

                # log progress
                # self.progress_bar.update(1)
                # self.progress_bar.set_description(f"Steps ({loss.item():.3f})")
            #
            # self.progress_bar.close()
            avg_loss = losses / count

            clean_test_accuracy = test_model(self.aggregator.model, self.aggregator.get_eval_dataloader(),
                                             self.aggregator.device)
            backdoor_asr = test_model(self.aggregator.model, self.backdoor_eval_dataloader, self.aggregator.device)

            print(f"Finished fine-tuning. Average loss: {avg_loss:.4f} | Benign accuracy: {clean_test_accuracy:.4f} | "
                  f"Backdoor ASR: {backdoor_asr:.4f}")

        
        return num_epochs, fine_tune_size, learning_rate, avg_loss, float(clean_test_accuracy.item()), float(backdoor_asr.item())

    def fine_tune_with_graph(self, pretrained_model=None, saved_logger=None, verbose=False):
        break_point = 10
        frequency = 4
        use_superfinetuning_hyper = True
        clip = self.aggregator_args.fine_tune_clip if self.aggregator_args.fine_tune_clip > 0 else torch.inf
        # clip = torch.inf
        if pretrained_model is not None:
            print(f"Loading model from {pretrained_model}")
            trained = torch.load(pretrained_model)
            self.aggregator.model.load_state_dict(trained)
        else:
            model = self.aggregator.model

        self.aggregator.model.to(self.aggregator.device)

        fine_tune_dataloader = self.aggregator.get_fine_tune_dataloader()
        self.aggregator.model.train()
        
        num_epochs = self.aggregator_args.fine_tune_epochs
        fine_tune_size = self.aggregator_args.fine_tune
        if use_superfinetuning_hyper:
            if self.env_args.dataset == "cifar10":
                lr_base = 0.0003
                lr_max1 = 0.1
                lr_max2 = 0.001
            elif self.env_args.dataset == "mnist":
                lr_base = 0.0006
                lr_max1 = 0.2
                lr_max2 = 0.002
            else:
                raise NotImplementedError
        else:
            if saved_logger is not None:
                # we base our lr based on what the model was trained on
                lr_base = saved_logger.progress.client_args.benign_lr/100
                lr_max1 = lr_base*300
                lr_max2 = lr_base*3

            else:
                lr_base = self.aggregator_args.fine_tune_lr/12
                lr_max1 = self.aggregator_args.fine_tune_lr
                lr_max2 = self.aggregator_args.fine_tune_lr/2

        optimizer = self.clients[0].get_optimizer(self.aggregator.model.parameters(),
                                                self.aggregator_args.fine_tune_optimizer,
                                                lr_base)

        save_folder = get_new_save_folder(os.path.join(self.env_args.save_path, "fine_tune/"))
        if clip != torch.inf:
            # self.env_args.model_file += f"_clip_{clip}"
            self.env_args.model_file = os.path.splitext(self.env_args.model_file)[0] + f"_clip_{clip}" + os.path.splitext(self.env_args.model_file)[1]
        logger = create_logger(save_folder)
        logger.set_epoch(0)
        logger.set_total_epochs(self.strategy_args.num_epochs)
        logger.set_config_args(
            self.strategy_args,
            self.client_args,
            self.aggregator_args,
            self.backdoor_args,
            self.env_args,
        )
        logger.save_progress()

        initial_clean_test_accuracy = test_model(self.aggregator.model, self.aggregator.get_eval_dataloader(), self.aggregator.device)
        initial_backdoor_asr = test_model(self.aggregator.model, self.backdoor_eval_dataloader, self.aggregator.device)
        print(f"Initial clean test accuracy {initial_clean_test_accuracy:.4f} | backdoor asr: {initial_backdoor_asr:.4f}")
        benign_accuracies = [float(initial_clean_test_accuracy.item())]
        backdoor_asrs = [float(initial_backdoor_asr.item())]

        print(f"Fine-tuning for {num_epochs} epochs with {fine_tune_size} samples")
        print(f"Using LRs: BASE: {lr_base}, MAX1: {lr_max1}, MAX2: {lr_max2} | Breakpoint: {break_point} | Frequency: {frequency}")

        # pbar = tqdm(range(num_epochs))
        for epoch in range(num_epochs):
            losses = 0.0
            count = 0
            lr_peak = lr_max1 if epoch < break_point else lr_max2
            phase = epoch % frequency
            current_lr = lr_base + (lr_peak - lr_base) * (1 - abs(2 * phase / frequency - 1))
            # if epoch < break_point:
            #     phase = epoch % frequency
            #     current_lr = lr_base + (lr_max1 - lr_base) * (1 - abs(2 * phase/frequency - 1))
            # else:
            #     # phase = (epoch - break_point) / break_point
            #     # current_lr = lr_base + (lr_max2 - lr_base) * (1 - abs(2 * phase - 1))
            #     phase = epoch % frequency
            #     current_lr = lr_base + (lr_max2 - lr_base) * (1 - abs(2 * phase/frequency - 1))


            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            pbar = tqdm(enumerate(fine_tune_dataloader))

            for _, (image, target) in pbar:
                image, target = image.to(self.aggregator.device), target.to(self.aggregator.device)

                # Forward
                output = self.aggregator.model(image)
                loss = F.cross_entropy(output, target)
                losses += float(loss.item()) * image.size(0)
                count += image.size(0)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                old_norm = torch.nn.utils.clip_grad_norm_(self.aggregator.model.parameters(), clip)
                new_norm = torch.nn.utils.clip_grad_norm_(self.aggregator.model.parameters(), clip)
                optimizer.step()
                acc = (output.argmax(1) == target).to(torch.float).mean()
                pbar.set_description(f"Epoch: {epoch+1} | Acc: {100*acc:.2f} | Old Grad Norm: {old_norm:.2e} | New Grad Norm: {new_norm:.2e}")
            avg_loss = losses / count
            pbar.close()
            clean_test_accuracy = test_model(self.aggregator.model, self.aggregator.get_eval_dataloader(),
                                            self.aggregator.device)
            backdoor_asr = test_model(self.aggregator.model, self.backdoor_eval_dataloader, self.aggregator.device)
            logger.update_epoch(epoch + 1, -1, clean_test_accuracy.item(), backdoor_asr.item())
            logger.save_progress(verbose=False)

            benign_accuracies.append(float(clean_test_accuracy.item()))
            backdoor_asrs.append(float(backdoor_asr.item()))

            print(f"Eval Epoch {epoch+1}/{num_epochs}: Learning Rate: {current_lr:.3e} | Loss: {avg_loss:.3e} | Benign Accuracy: {clean_test_accuracy:.4f} | "
                f"Backdoor ASR: {backdoor_asr:.4f}")

        logger.set_done()  # set progress to "finished"

        if verbose:
            plt.figure(figsize=(8, 5))
            plt.plot(range(num_epochs + 1), benign_accuracies, label="Benign Accuracy", marker="o", linestyle="-")
            plt.plot(range(num_epochs + 1), backdoor_asrs, label="Backdoor ASR", marker="s", linestyle="--")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title(f"Fine-tuning Results for Norm Bound: 2, Epochs: {num_epochs}, Fine Tuning Dataset Size: {fine_tune_size}, LR: {self.aggregator_args.fine_tune_lr}")
            plt.legend()
            plt.grid(True)

            filename = f"super_norm_bound2_{fine_tune_size}_{num_epochs}_{self.aggregator_args.fine_tune_lr}_adam.png"
            plt.savefig(filename)
            print(f"Plot saved as {filename}")
    
        return num_epochs, fine_tune_size, self.aggregator_args.fine_tune_lr, avg_loss, float(clean_test_accuracy.item()), float(backdoor_asr.item())
