import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masks import get_conditional_pixels
from torchvision.utils import make_grid


class Trainer():
    """Class used to train PixelCNN models without conditioning.

    Parameters
    ----------
    model : pixconcnn.models.gated_pixelcnn.GatedPixelCNN(RGB) instance

    optimizer : one of optimizers in torch.optim

    device : torch.device instance

    record_loss_every : int
        Frequency (in iterations) with which to record loss.

    save_model_every : int
        Frequency (in epochs) with which to save model.
    """
    def __init__(self, model, optimizer, device, record_loss_every=10,
                 save_model_every=5):
        self.device = device
        self.losses = {'total': []}
        self.mean_epoch_losses = []
        self.model = model
        self.optimizer = optimizer
        self.record_loss_every = record_loss_every
        self.save_model_every = save_model_every
        self.steps = 0

    def train(self, data_loader, epochs, directory='.'):
        """Trains model on the data given in data_loader.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader instance

        epochs : int
            Number of epochs to train model for.

        directory : string
            Directory in which to store training progress, including trained
            models and samples generated at every epoch.

        Returns
        -------
        List of numpy arrays of generated images after each epoch.
        """
        # List of generated images after each epoch to track progress of model
        progress_imgs = []

        for epoch in range(epochs):
            print("\nEpoch {}/{}".format(epoch + 1, epochs))
            epoch_loss = self._train_epoch(data_loader)
            mean_epoch_loss = epoch_loss / len(data_loader)
            print("Epoch loss: {}".format(mean_epoch_loss))
            self.mean_epoch_losses.append(mean_epoch_loss)

            # Create a grid of model samples (limit number of samples by scaling
            # by number of pixels in output image; this is needed because of
            # GPU memory limitations)
            if self.model.img_size[-1] > 32:
                scale_to_32 = self.model.img_size[-1] / 32
                num_images = 64 / (scale_to_32 * scale_to_32)
            else:
                num_images = 64
            # Generate samples from model
            if epoch % 10 == 0:
                samples = self.model.sample(self.device, num_images)
                img_grid = make_grid(samples).cpu()
                # Convert to numpy with channels in imageio order
                img_grid = img_grid.float().numpy().transpose(1, 2, 0) / (self.model.num_colors - 1.)
                progress_imgs.append(img_grid)
                # Save generated image
                imageio.imsave(directory + '/training{}.png'.format(epoch), progress_imgs[-1])
                # Save model
            if epoch % self.save_model_every == 0:
                torch.save(self.model.state_dict(),
                           directory + '/model{}.pt'.format(epoch))

        return progress_imgs

    def _train_epoch(self, data_loader):
        epoch_loss = 0
        for i, (batch, _) in enumerate(data_loader):
            batch_loss = self._train_iteration(batch)
            epoch_loss += batch_loss
            if i % 50 == 0:
                print("Iteration {}/{}, Loss: {}".format(i + 1,
                                                         len(data_loader),
                                                         batch_loss))
        return epoch_loss

    def _train_iteration(self, batch):
        self.optimizer.zero_grad()

        batch = batch.to(self.device)

        # Normalize batch, i.e. put it in 0 - 1 range before passing it through
        # the model
        norm_batch = batch.float() / (self.model.num_colors - 1)
        logits = self.model(norm_batch)

        loss = self._loss(logits, batch)
        loss.backward()
        self.optimizer.step()

        self.steps += 1


        return loss.item()

    def _loss(self, logits, batch):
        loss = F.cross_entropy(logits, batch)

        if self.steps % self.record_loss_every == 0:
            self.losses['total'].append(loss.item())

        return loss

class Trainer_3D():
    """Class used to train PixelCNN models without conditioning.

    Parameters
    ----------
    model : pixconcnn.models.gated_pixelcnn.GatedPixelCNN(RGB) instance

    optimizer : one of optimizers in torch.optim

    device : torch.device instance

    record_loss_every : int
        Frequency (in iterations) with which to record loss.

    save_model_every : int
        Frequency (in epochs) with which to save model.
    """
    def __init__(self, model, optimizer, device, record_loss_every=10,
                 save_model_every=5):
        self.device = device
        self.losses = {'total': []}
        self.mean_epoch_losses = []
        self.model = model
        self.optimizer = optimizer
        self.record_loss_every = record_loss_every
        self.save_model_every = save_model_every
        self.steps = 0

    def train(self, data_loader, epochs, directory='.'):
        """Trains model on the data given in data_loader.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader instance

        epochs : int
            Number of epochs to train model for.

        directory : string
            Directory in which to store training progress, including trained
            models and samples generated at every epoch.

        Returns
        -------
        List of numpy arrays of generated images after each epoch.
        """
        # List of generated images after each epoch to track progress of model
        progress_imgs = []

        for epoch in range(1,epochs+1):
            print("\nEpoch {}/{}".format(epoch + 1, epochs))
            epoch_loss = self._train_epoch(data_loader)
            mean_epoch_loss = epoch_loss / len(data_loader)
            print("Epoch loss: {}".format(mean_epoch_loss))
            self.mean_epoch_losses.append(mean_epoch_loss)

            # Create a grid of model samples (limit number of samples by scaling
            # by number of pixels in output image; this is needed because of
            # GPU memory limitations)
            if self.model.img_size[-1] > 32:
                scale_to_32 = self.model.img_size[-1] / 32
                num_images = 64 / (scale_to_32 * scale_to_32)
            else:
                num_images = 64
            # Generate samples from model
            if epoch % 10 == 0:
                samples = self.model.sample(self.device, num_images)
                img_grid = make_grid(samples).cpu()
                # Convert to numpy with channels in imageio order
                img_grid = img_grid.uint8().numpy().transpose(1, 2, 0)
                progress_imgs.append(img_grid)
                # Save generated image
                imageio.imsave(directory + '/training{}.png'.format(epoch), progress_imgs[-1])
                # Save model
            if epoch % self.save_model_every == 0:
                torch.save(self.model.state_dict(),
                           directory + '/model{}.pt'.format(epoch))

        return progress_imgs

    def _train_epoch(self, data_loader):
        epoch_loss = 0
        for i, (batch, _) in enumerate(data_loader):
            batch_loss = self._train_iteration(batch)
            epoch_loss += batch_loss
            if i % 50 == 0:
                print("Iteration {}/{}, Loss: {}".format(i + 1,
                                                         len(data_loader),
                                                         batch_loss))
        return epoch_loss

    def _train_iteration(self, batch):
        self.optimizer.zero_grad()

        batch = batch.to(self.device)

        # Normalize batch, i.e. put it in 0 - 1 range before passing it through
        # the model
        norm_batch = batch.float() / (self.model.num_colors - 1)
        logits = self.model(norm_batch)

        loss = self._loss(logits, batch)
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        return loss.item()

    def _loss(self, logits, batch):
        loss = F.cross_entropy(logits, batch)

        if self.steps % self.record_loss_every == 0:
            self.losses['total'].append(loss.item())

        return loss

