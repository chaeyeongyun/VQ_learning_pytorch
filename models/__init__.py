from torch import nn
from .backbones import ResNet50, VGG16
from .decoder import Decoder_x3
from vector_quantizer.vq_img import VectorQuantizer

backbone_dict = {
    'resnet50':(ResNet50, 3, 2048),
    'vgg16':(VGG16, 2, 512)
}


class Network(nn.Module):
    def __init__(self, backbone, vq_cfg) :
        super().__init__()
        backbone_cls, num_upsample, out_channels = backbone_dict[backbone]
        self.backbone = backbone_cls()
        if num_upsample == 3:
            self.decoder = Decoder_x3(in_channels=out_channels)
        else:
            raise NotImplementedError
        self.vq = VectorQuantizer(**vq_cfg) 
    def forward(self, x):
        output = self.backbone(x)
        quantize, embed_index, loss = self.vq(output)
        output = self.decoder(output)
        return output, loss
    