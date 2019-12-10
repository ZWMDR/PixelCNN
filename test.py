import numpy as np
import imageio
from PIL import Image  # PIL pakage name is Pillow


path = '/home/luoty/code_python/PixelCNN/2019-12-11_01-07_cifar10_model/'
img = imageio.imread(path+'training10.png')
print(img[:][:][2])
high, width, ichannel = img.shape
print(type(img))
print(img.shape)
imageio.imwrite(path+'leopard_i1.jpg', img)
imageio.imwrite(path+'leopard_i2.jpg', np.float32(img / 10))  # automatic brightness adjust
imageio.imwrite(path+'leopard_i3.jpg', np.uint8(img / 10))


