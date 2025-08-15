import tqdm

from src.arguments.client_args import ClientArgs
from src.arguments.env_args import EnvArgs
from src.arguments.backdoor_args import BackdoorArgs
from src.utils.logging import Logger
from src.clients.static import StaticBackdoorClient
import torch.nn.functional as F
import torch
from torch.utils.data.dataloader import DataLoader


class NeurotoxinClient(StaticBackdoorClient):
    def __init__(
            self,
            client_args: ClientArgs,
            backdoor_args: BackdoorArgs,
            env_args: EnvArgs = None,
            logger: Logger = None,
    ):
        super(NeurotoxinClient, self).__init__(client_args, backdoor_args, env_args, logger)
        self.mask_grad_list = []

    def _epoch_fit(self):
        # we need to change it to apply the gradient mask like they do
        self.model.train()
        losses = 0.0
        count = 0
        for _, (image, target) in enumerate(self.dataloader):
            # yeet features and labels to gpu
            image, target = image.to(self.device), target.to(self.device)

            # forward
            output = self.model(image)
            loss = F.cross_entropy(output, target)
            losses += float(loss.item()) * image.size(0)
            count += image.size(0)
            self.optimizer.zero_grad()

            # back prop
            loss.backward()
            if self.client_args.neurotoxin_mask_ratio != 1:
                self.apply_grad_mask()

            # update
            self.optimizer.step()

            # log progress
            self.progress_bar.update(1)
            self.progress_bar.set_description(f"Steps ({loss.item():.3f})")

        self.progress_bar.close()
        self.loss = losses / count


    def compute_grad_mask(self, model_state_dict):
        print(f"[Neurotoxin] Computing gradient mask")
        original_state_dict = self.model.state_dict()
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        clean_loader = DataLoader(self.clean_dataset, shuffle=False, batch_size=128, num_workers=16)
        mask_grad_list = []
        for inputs, labels in clean_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            output = self.model(inputs)

            loss = F.cross_entropy(output, labels)
            loss.backward(retain_graph=True)

        k_layer = 0
        for _, parms in self.model.named_parameters():
            if parms.requires_grad:
                gradients = parms.grad.abs().view(-1)
                gradients_length = len(gradients)
                if self.client_args.neurotoxin_mask_ratio == 1.0:
                    _, indices = torch.topk(-1 * gradients, int(gradients_length * 1.0))
                else:
                    _, indices = torch.topk(-1 * gradients, int(gradients_length * self.client_args.neurotoxin_mask_ratio))

                mask_flat = torch.zeros(gradients_length)
                mask_flat[indices.cpu()] = 1.0
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                k_layer += 1

        # undo what we did
        self.model.load_state_dict(original_state_dict)
        self.mask_grad_list = mask_grad_list

    def apply_grad_mask(self):
        # print(f"[Neurotoxin] Apply gradient mask")
        mask_grad_list_copy = iter(self.mask_grad_list)
        for _, parms in self.model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)

    def train_one_epoch(self, model_state_dict, benign_states, *args, **kwargs):
        if self.backdoor_args.refresh_backdoor:
            assert self.original_dataset is not None, f"Original dataset is None, it means the dataset was not set before training"
            self.set_dataset(self.original_dataset.copy())

        if kwargs.get("no_backdoor", None) is not None and kwargs["no_backdoor"] == True:
            self.set_dataset(self.original_dataset.copy(), no_backdoor=True)

        self._setup_fit(model_state_dict)

        self.compute_grad_mask(model_state_dict)

        for i in range(self.client_args.malicious_epoch):
            self.progress_bar = tqdm.tqdm(total=len(self.dataloader), desc="Steps", position=0, leave=True)
            self._local_eval()
            self._epoch_fit()
            self._local_eval()

        self._log_epoch(self.clean_test_accuracy, self.loss, metadata={}, backdoor_accuracy=self.backdoor_asr)

        model = self.model.state_dict()
        for k in model.keys():
            # do this because mal_epoch can be different from benign_epoch
            # want all model updates to share the same num_batches_tracked
            # this is not a problem when mal_epoch == benign_epoch
            if "num_batches_tracked" in k:
                model[k] = model_state_dict[k] + self.client_args.benign_epoch * len(self.dataloader)
                # model[k] = model_state_dict[k] * (self.client_args.malicious_epoch / self.client_args.benign_epoch)
        if kwargs.get("no_backdoor", None) is not None and kwargs["no_backdoor"] == True:
            # we reset back to using the backdoor
            self.set_dataset(self.original_dataset.copy(), no_backdoor=False)

        return model
