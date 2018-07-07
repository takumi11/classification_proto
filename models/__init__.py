from .four_lays_cnn import FOURLAYSCNN
from .myresnet import MyResNet
from .myvgg import MyVGG
from .pre_four_lays_cnn import PreFOURLAYSCNN

archs = {
        '4-lay': FOURLAYSCNN,
        'pre-4-lay': PreFOURLAYSCNN,
        'my-vgg': MyVGG,
        'my-resnet': MyResNet
        }
