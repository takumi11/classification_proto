from .four_lays_cnn import FOURLAYSCNN
from .myresnet import MyResNet

archs = {
        '4-lay': FOURLAYSCNN,
        'my-resnet': MyResNet
    }
