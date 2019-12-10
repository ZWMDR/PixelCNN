import torch
import torch.nn as nn


class MaskedConv2d(nn.Conv2d):
    """
    Implements various 2d masked convolutions.

    Parameters
    ----------
    mask_type : string
        Defines the type of mask to use. One of 'A', 'A_Red', 'A_Green',
        'A_Blue', 'B', 'B_Red', 'B_Green', 'B_Blue', 'H', 'H_Red', 'H_Green',
        'H_Blue', 'HR', 'HR_Red', 'HR_Green', 'HR_Blue', 'V', 'VR'.
    """

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.mask_type = mask_type

        # Initialize mask
        mask = torch.zeros(*self.weight.size())
        _, kernel_c, kernel_h, kernel_w = self.weight.size()
        # If using a color mask, the number of channels must be divisible by 3
        if mask_type.endswith('Red') or mask_type.endswith('Green') or mask_type.endswith('Blue'):
            assert kernel_c % 3 == 0
        # If using a horizontal mask, the kernel height must be 1
        if mask_type.startswith('H'):
            assert kernel_h == 1  # Kernel should have shape (1, kernel_w)

        if mask_type == 'A':
            # For 3 by 3 kernel, this would be:
            #  1 1 1
            #  1 0 0
            #  0 0 0
            mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.
            mask[:, :, :kernel_h // 2, :] = 1.
        elif mask_type == 'A_Red':
            # Mask type A for red channels. Same as regular mask A
            if kernel_h == 1 and kernel_w == 1:
                pass  # Mask is all zeros for 1x1 convolution
            else:
                mask[:, :kernel_c // 3, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :kernel_c // 3, :kernel_h // 2, :] = 1.
        elif mask_type == 'A_Green':
            # Mask type A for green channels. Same as regular mask A, except
            # central pixel of first third of channels is 1
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :kernel_c // 3, 0, 0] = 1.
            else:
                mask[:, :kernel_c * 2 // 3, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :kernel_c * 2 // 3, :kernel_h // 2, :] = 1.
                mask[:, :kernel_c // 3, kernel_h // 2, kernel_w // 2 + 1] = 1.
        elif mask_type == 'A_Blue':
            # Mask type A for blue channels. Same as regular mask A, except
            # central pixel of first two thirds of channels is 1
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :2 * kernel_c // 3, 0, 0] = 1.
            else:
                mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :kernel_h // 2, :] = 1.
                mask[:, :2 * kernel_c // 3, kernel_h // 2, kernel_w // 2 + 1] = 1.
        elif mask_type == 'B':
            # For 3 by 3 kernel, this would be:
            #  1 1 1
            #  1 1 0
            #  0 0 0
            mask[:, :, :kernel_h // 2, :] = 1.
            mask[:, :, kernel_h // 2, :kernel_w // 2 + 1] = 1.
        elif mask_type == 'B_Red':
            # Mask type B for red channels. Same as regular mask B, except last
            # two thirds of channels of central pixels are 0. Alternatively,
            # same as Mask A but with first third of channels of central pixels
            # are 1
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :kernel_c // 3, 0, 0] = 1.
            else:
                mask[:, :kernel_c // 3, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :kernel_c // 3, :kernel_h // 2, :] = 1.
                mask[:, :kernel_c // 3, kernel_h // 2, kernel_w // 2 + 1] = 1.
        elif mask_type == 'B_Green':
            # Mask type B for green channels. Same as regular mask B, except
            # last third of channels of central pixels are 0
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :2 * kernel_c // 3, 0, 0] = 1.
            else:
                mask[:, :kernel_c * 2 // 3, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :kernel_c * 2 // 3, :kernel_h // 2, :] = 1.
                mask[:, :2 * kernel_c // 3, kernel_h // 2, kernel_w // 2 + 1] = 1.
        elif mask_type == 'B_Blue':
            # Mask type B for blue channels. Same as regular mask B
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :, 0, 0] = 1.
            else:
                mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :kernel_h // 2, :] = 1.
                mask[:, :, kernel_h // 2, kernel_w // 2 + 1] = 1.

        # Register buffer adds a key to the state dict of the model. This will
        # track the attribute without registering it as a learnable parameter.
        # We require this since mask will be used in the forward pass.
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class MaskedConv3d(nn.Conv3d):
    """
    Implements various 2d masked convolutions.

    Parameters
    ----------
    mask_type : string
        Defines the type of mask to use. One of 'A', 'A_Red', 'A_Green',
        'A_Blue', 'B', 'B_Red', 'B_Green', 'B_Blue', 'H', 'H_Red', 'H_Green',
        'H_Blue', 'HR', 'HR_Red', 'HR_Green', 'HR_Blue', 'V', 'VR'.
    """

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)
        self.mask_type = mask_type

        # Initialize mask
        mask = torch.zeros(*self.weight.size())
        _, kernel_c, kernel_d, kernel_h, kernel_w = self.weight.size()
        # If using a color mask, the number of channels must be divisible by 3
        if mask_type.endswith('Red') or mask_type.endswith('Green') or mask_type.endswith('Blue'):
            assert kernel_c % 3 == 0
        # If using a horizontal mask, the kernel height must be 1
        if mask_type == 'A':
            # For 3 by 3 kernel, this would be:
            #  1 1 1
            #  1 0 0
            #  0 0 0
            mask[:, :, :, kernel_h // 2, :kernel_w // 2] = 1.
            mask[:, :, :, :kernel_h // 2, :] = 1.
        elif mask_type == 'A_Red':
            # Mask type A for red channels. Same as regular mask A
            if kernel_h == 1 and kernel_w == 1:
                pass  # Mask is all zeros for 1x1 convolution
            else:
                mask[:, :, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :, :kernel_h // 2, :] = 1.
        elif mask_type == 'A_Green':
            # Mask type A for green channels. Same as regular mask A, except
            # central pixel of first third of channels is 1
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :, :, kernel_c // 3, 0, 0] = 1.
            else:
                mask[:, :, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :, :kernel_h // 2, :] = 1.
                mask[:, :kernel_c // 3, :, kernel_h // 2, kernel_w // 2] = 1.
        elif mask_type == 'A_Blue':
            # Mask type A for blue channels. Same as regular mask A, except
            # central pixel of first two thirds of channels is 1
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :2 * kernel_c // 3, :, 0, 0] = 1.
            else:
                mask[:, :, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :, :kernel_h // 2, :] = 1.
                mask[:, :2 * kernel_c // 3, :, kernel_h // 2, kernel_w // 2] = 1.
        elif mask_type == 'B':
            # For 3 by 3 kernel, this would be:
            #  1 1 1
            #  1 1 0
            #  0 0 0
            mask[:, :, :, kernel_h // 2, :kernel_w // 2 + 1] = 1.
            mask[:, :, :, :kernel_h // 2, :] = 1.
        elif mask_type == 'B_Red':
            # Mask type B for red channels. Same as regular mask B, except last
            # two thirds of channels of central pixels are 0. Alternatively,
            # same as Mask A but with first third of channels of central pixels
            # are 1
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :kernel_c // 3, :, 0, 0] = 1.
            else:
                mask[:, :, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :, :kernel_h // 2, :] = 1.
                mask[:, :kernel_c // 3, :, kernel_h // 2, kernel_w // 2] = 1.
        elif mask_type == 'B_Green':
            # Mask type B for green channels. Same as regular mask B, except
            # last third of channels of central pixels are 0
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :2 * kernel_c // 3, :, 0, 0] = 1.
            else:
                mask[:, :, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :, :kernel_h // 2, :] = 1.
                mask[:, :2 * kernel_c // 3, :, kernel_h // 2, kernel_w // 2] = 1.
        elif mask_type == 'B_Blue':
            # Mask type B for blue channels. Same as regular mask B
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :, :, 0, 0] = 1.
            else:
                mask[:, :, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :, :kernel_h // 2, :] = 1.
                mask[:, :, :, kernel_h // 2, kernel_w // 2] = 1.

        # Register buffer adds a key to the state dict of the model. This will
        # track the attribute without registering it as a learnable parameter.
        # We require this since mask will be used in the forward pass.
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv3d, self).forward(x)


class MaskedConvRGB(nn.Module):
    """
    Masked convolution with RGB channel splitting.

    Parameters
    ----------
    mask_type : string
        One of 'A', 'B', 'V' or 'H'.

    in_channels : int
        Must be divisible by 3

    out_channels : int
        Must be divisible by 3

    kernel_size : int or tuple of ints

    stride : int

    padding : int

    bias : bool
        If True adds a bias term to the convolution.
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size,
                 stride, padding, bias):
        super(MaskedConvRGB, self).__init__()

        self.conv_R = MaskedConv2d(mask_type + '_Red', in_channels,
                                   out_channels // 3, kernel_size,
                                   stride=stride, padding=padding, bias=bias)
        self.conv_G = MaskedConv2d(mask_type + '_Green', in_channels,
                                   out_channels // 3, kernel_size,
                                   stride=stride, padding=padding, bias=bias)
        self.conv_B = MaskedConv2d(mask_type + '_Blue', in_channels,
                                   out_channels // 3, kernel_size,
                                   stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out_red = self.conv_R(x)
        out_green = self.conv_G(x)
        out_blue = self.conv_B(x)
        return torch.cat([out_red, out_green, out_blue], dim=1)
