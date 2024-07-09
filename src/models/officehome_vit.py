import torch.nn as nn
from ..models import network
import timm


class OfficeHome_ViT(nn.Module):
    def __init__(self):
        super(OfficeHome_ViT, self).__init__()
        self.netF = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.netB = network.feat_bottleneck(type='bn', feature_dim=768,
                                            bottleneck_dim=256)
        self.netC = network.feat_classifier(type='wn', class_num=65, bottleneck_dim=256)
        self._num_classes = 65
        self._output_dim = 256
        pass

    def forward(self, x, return_feats=False):
        feats = self.netF.forward_features(x)
        feats = self.netF.fc_norm(feats[:, 0])
        feats = self.netB(feats)
        outputs = self.netC(feats)
        if return_feats:
            return feats, outputs
        else:
            return outputs
