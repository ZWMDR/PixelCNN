from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import sys
sys.path.append('/home/luoty/code_python/PixelCNN/utils')
from MyDatasets import MyDatasets
def mnist(batch_size=128, num_colors=256, size=28,
          path_to_data='../mnist_data'):
    """MNIST dataloader with (28, 28) images.

    Parameters
    ----------
    batch_size : int

    num_colors : int
        Number of colors to quantize images into. Typically 256, but can be
        lower for e.g. binary images.

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    quantize = get_quantize_func(num_colors)

    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: quantize(x))
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def CIFAR10(batch_size=128, type = 0,num_colors=256, size=28,
          path_to_data='./cifar10_data'):
    """MNIST dataloader with (28, 28) images.

    Parameters
    ----------
    batch_size : int

    num_colors : int
        Number of colors to quantize images into. Typically 256, but can be
        lower for e.g. binary images.

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    quantize = get_quantize_func(num_colors)

    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: quantize(x))
    ])

    train_data = MyDatasets(path_to_data, type = 0,train=True, download=False,
                                transform=all_transforms)
    test_data = MyDatasets(path_to_data,type = 0, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
def get_quantize_func(num_colors):
    """Returns a quantization function which can be used to set the number of
    colors in an image.

    Parameters
    ----------
    num_colors : int
        Number of bins to quantize image into. Should be between 2 and 256.
    """
    def quantize_func(batch):
        """Takes as input a float tensor with values in the 0 - 1 range and
        outputs a long tensor with integer values corresponding to each
        quantization bin.

        Parameters
        ----------
        batch : torch.Tensor
            Values in 0 - 1 range.
        """
        if num_colors == 2:
            return (batch > 0.5).long()
        else:
            return (batch * (num_colors - 1)).long()

    return quantize_func

