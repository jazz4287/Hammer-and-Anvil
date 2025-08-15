import torch
import torch.nn.functional as F
import numpy as np

import random
import uuid
import os
import os.path as path


# static seeding
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  # for determinism
    torch.use_deterministic_algorithms(True)


# get a temp save folder for the model
def get_new_save_folder(save_path="./models"):
    run_uuid = uuid.uuid4().hex
    save_folder = path.join(
        save_path,
        run_uuid[0:5],
    )

    # create save folder
    try:
        if not path.exists(save_folder):
            print(f"Creating save folder at {save_folder}.")
            os.makedirs(save_folder)
    except Exception as e:
        print(f"Error creating save folder: {e}")
        raise e

    return save_folder


# test pytorch model using loader
def test_model(model, loader, device, data_transform=None):
    # print(f"Loaded {len(loader)} samples.")
    model.eval()
    corrent_predictions, total_predictions = 0, 0
    with torch.no_grad():
        for _, (image, target) in enumerate(loader):
            if data_transform is not None:
                image, target = data_transform(image, target)
            image, target = image.to(device), target.to(device)

            output = model(image)
            predicted = output.argmax(dim=1)
            total_predictions += target.size(0)
            corrent_predictions += (predicted == target).to(torch.int).sum()

    # print(f"tot {total_predictions}  correct {corrent_predictions.float()}")
    return corrent_predictions.float() / total_predictions


def set_model_params(orig_model_params, new_model_params):
    """
    From a new set of model_params, sets the .data attributes of the existing model parameters without creating
    new parameter objects
    :param orig_model_params: existing model params into which the new .data will be copied into (preserves original
    pointers and parameter objects).
    :param new_model_params: new model parameter dictionary.
    :return:
    """
    with torch.no_grad():
        for orig_param, new_param in zip(
            orig_model_params.values(), new_model_params.values()
        ):
            # orig_param.copy_(new_param)
            orig_param.data = new_param.data
            # orig_model_params[key].data.copy_(torch.zeros_like(new_model_params[key].data))
        # for orig_param, new_param in zip(orig_model_params.values(), new_model_params.values()):
        #     orig_param.data.copy_(new_param.data)


def get_model_params(model):
    """
    Custom method for getting pointers to the model parameters rather than a state_dict copy to allow for a direct
    modification on device of the weights by accessing the .data attribute Tensor.
    :return: a dictionary where keys are numbers (0 to number of parameter objects - 1) and the values stored are
    parameters (same parameters are the ones used by the model, should not be replaced but instead the .data
    attribute should be modified instead)
    """
    model_params = {}
    for i, m in enumerate(model.parameters()):
        model_params[i] = m
    return model_params
