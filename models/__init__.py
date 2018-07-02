from .four_lays_cnn import FOURLAYSCNN
from .myresnet import MyResNet
from .pre_four_lays_cnn import PreFOURLAYSCNN

archs = {
        '4-lay': FOURLAYSCNN,
        'pre-4-lay': PreFOURLAYSCNN,
        'my-resnet': MyResNet
        }
