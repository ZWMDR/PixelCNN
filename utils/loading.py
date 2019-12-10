import json
import torch
from utils.dataloaders import mnist, CIFAR10
from utils.init_models import initialize_model

def load_model(directory, model_version=None):
    """
    Returns model, data_loader and mask_descriptor of trained model.

    Parameters
    ----------
    directory : string
        Directory where experiment was saved. For example './experiment_1'.

    model_version : int or None
        If None loads final model, otherwise loads model version determined by
        int.
    """
    path_to_config = directory + '/config.json'
    if model_version is None:
        path_to_model = directory + '/model.pt'
    else:
        path_to_model = directory + '/model{}.pt'.format(model_version)

    # Open config file
    with open(path_to_config) as config_file:
        config = json.load(config_file)

    # Load dataset info
    dataset = config["dataset"]
    resize = config["resize"]
    batch_size = config["batch_size"]
    num_colors = config["num_colors"]


    # Get data
    if dataset == 'mnist':
        # Extract the test dataset (second argument)
        _, data_loader = mnist(batch_size, num_colors, resize)
        img_size = (1, resize, resize)
    elif dataset == 'cifar10':
        _, data_loader = CIFAR10(batch_size, num_colors=num_colors, size=resize)
        img_size = (3, resize, resize)

    # Load model info
    depth = config["depth"]
    num_filters_prior = config["num_filters_prior"]
    filter_size = config["filter_size"]

    model = initialize_model(img_size,
                             num_colors,
                             depth,
                             filter_size,
                             num_filters_prior)

    model.load_state_dict(torch.load(path_to_model, map_location=lambda storage, loc: storage))

    return model, data_loader
