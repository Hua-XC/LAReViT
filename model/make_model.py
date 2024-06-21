import copy

from model.vision_transformer import ViT
import torch
import torch.nn as nn



class PositionalChannelReconstructionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(PositionalChannelReconstructionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, _ = x.size()
        y = (1*self.avg_pool(x)+0*self.max_pool(x)).view(batch_size, num_channels)
        y = self.fc(y).view(batch_size, num_channels, 1)
        return x * y.expand_as(x)
 
# L2 norm
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_vision_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer, self).__init__()
        self.in_planes = 768

        self.base = ViT(img_size=[cfg.H,cfg.W],
                        stride_size=cfg.STRIDE_SIZE,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)
        self.baseA = ViT(img_size=[cfg.H,cfg.W],
                        stride_size=[10,10],
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)

        self.baseA.load_param(cfg.PRETRAIN_PATH)
        #print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        layer_norm = self.base.norm
        self.layer_norm0 = copy.deepcopy(layer_norm)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


        self.classifier1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier1.apply(weights_init_classifier)
        layer_norm = self.base.norm
        self.layer_norm1 = copy.deepcopy(layer_norm)

        self.bottleneck1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)
        
        self.classifier2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)
        layer_norm = self.base.norm
        self.layer_norm2 = copy.deepcopy(layer_norm)
    
        self.bottleneck2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)
        
        self.classifier3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier3.apply(weights_init_classifier)
        layer_norm = self.base.norm
        self.layer_norm3 = copy.deepcopy(layer_norm)

        self.bottleneck3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck3.bias.requires_grad_(False)
        self.bottleneck3.apply(weights_init_kaiming)
        
        
        self.l2norm = Normalize(2)

        self.PCR = PositionalChannelReconstructionModule(364)
    def forward(self, x):
        
        x = self.base.patch_embed(x)
        
        x = x.flatten(2).transpose(1, 2)
        x = self.PCR(x) + x
        
        x,x_1,x_2,x_3,x_s = self.base(x)
        x   = self.layer_norm1(x  )
        x_1 = self.layer_norm1(x_1)
        x_2 = self.layer_norm1(x_2)
        x_3 = self.layer_norm1(x_3)
        x_s = self.layer_norm2(x_s)

        
        feat =    self.bottleneck1(x  )
        feat_x1 = self.bottleneck1(x_1)
        feat_x2 = self.bottleneck1(x_2)
        feat_x3 = self.bottleneck1(x_3)
        feat_x_s = self.bottleneck2(x_s)

        if self.training:

            cls_score = self.classifier1(feat)
            cls_score1 = self.classifier1(feat_x1)
            cls_score2 = self.classifier1(feat_x2)
            cls_score3 = self.classifier1(feat_x3)
            cls_score_s = self.classifier2(feat_x_s)
   
            return cls_score, cls_score1,cls_score2,cls_score3, \
                   x,x_1,x_2,x_3,cls_score_s,x_s


        else:
            return self.l2norm(feat),self.l2norm(x_s)#,self.l2norm(feat_x1),self.l2norm(feat_x2),self.l2norm(feat_x3)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))