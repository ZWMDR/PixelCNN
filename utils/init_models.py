from pixconcnn.models.gated_pixelcnn import PixelCNN, PixelCNNRGB ,PixelCNN_3D
def initialize_model(img_size, num_colors, depth, filter_size,
                     num_filters_prior,type = '2D'):
    """Helper function that initializes an appropriate model based on the
    input arguments.

    Parameters
    ----------
    img_size : tuple of ints
        Specifies size of image as (channels, height, width), e.g. (3, 32, 32).
        If img_size[0] == 1, returned model will be for grayscale images. If
        img_size[0] == 3, returned model will be for RGB images.

    num_colors : int
        Number of colors to quantize output into. Typically 256, but can be
        lower for e.g. binary images.

    depth : int
        Number of layers in model.

    filter_size : int
        Size of (square) convolutional filters of the model.

    constrained : bool
        If True returns a PixelConstrained model, otherwise returns a
        GatedPixelCNN or GatedPixelCNNRGB model.

    num_filters_prior : int
        Number of convolutional filters in each layer of prior network.

    num_filter_cond : int (optional)
        Required if using a PixelConstrained model. Number of of convolutional
        filters in each layer of conditioning network.
    """
    if type == '2D':
        if img_size[0] == 1:
            prior_net = PixelCNN(img_size=img_size,
                                      num_colors=num_colors,
                                      num_filters=num_filters_prior,
                                      depth=depth,
                                      filter_size=filter_size)
        else:
            prior_net = PixelCNNRGB(img_size=img_size,
                                         num_colors=num_colors,
                                         num_filters=num_filters_prior,
                                         depth=depth,
                                         filter_size=filter_size)
    else:
        if img_size[0] == 1:
            prior_net = PixelCNN_3D(img_size=img_size,
                                      num_colors=num_colors,
                                      num_filters=num_filters_prior,
                                      depth=depth,
                                      filter_size=filter_size)
        # else:
        #     prior_net = PixelCNNRGB_3D(img_size=img_size,
        #                                  num_colors=num_colors,
        #                                  num_filters=num_filters_prior,
        #                                  depth=depth,
        #                                  filter_size=filter_size)



    return prior_net
