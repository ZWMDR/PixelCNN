import imageio
import json
import os
import sys
import time
import torch
from pixconcnn.training import Trainer
from utils.dataloaders import mnist, CIFAR10
from utils.init_models import initialize_model

# Set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main.py <path_to_config>"))
config_path = sys.argv[1]

# Open config file
with open(config_path) as config_file:
    config = json.load(config_file)

name = config['name']
batch_size = config['batch_size']
lr = config['lr']
type = config['type']
num_colors = config['num_colors']
epochs = config['epochs']
dataset = config['dataset']
resize = config['resize']  # Only relevant for celeba
filter_size = config['filter_size']
depth = config['depth']
num_filters_prior = config['num_filters_prior']


# Create a folder to store experiment results
timestamp = time.strftime("%Y-%m-%d_%H-%M")
directory = "{}_{}".format(timestamp, name)
if not os.path.exists(directory):
    os.makedirs(directory)

# Save config file in experiment directory
with open(directory + '/config.json', 'w') as config_file:
    json.dump(config, config_file)

# Get data
if dataset == 'mnist':
    data_loader, _ = mnist(batch_size, num_colors=num_colors, size=resize)
    img_size = (1, resize, resize)
elif dataset == 'cifar10':
    data_loader, _ = CIFAR10(batch_size, num_colors=num_colors, size=resize)
    img_size = (3, resize, resize)
# Initialize model weights and architecture


model = initialize_model(img_size,
                         num_colors,
                         depth,
                         filter_size,
                         num_filters_prior,
                         type)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
trainer = Trainer(model, optimizer, device)
progress_imgs = trainer.train(data_loader, epochs, directory=directory)

# Save losses and plots of them
with open(directory + '/losses.json', 'w') as losses_file:
    json.dump(trainer.losses, losses_file)

# Save model
torch.save(trainer.model.state_dict(), directory + '/model.pt')

# Save gif of progress
imageio.mimsave(directory + '/training.gif', progress_imgs, fps=24)
