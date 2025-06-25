from numpy.lib._function_base_impl import extract
import torch
import torch.nn as nn

from . import cnn_backbones
from omegaconf import OmegaConf



class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()
        self.cfg = cfg

        self.output_dim = cfg.model.text.embedding_dim
        
        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)
        self.local_embedder = nn.Conv2d(
            self.interm_feature_dim,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if cfg.model.vision.freeze_cnn:
            print("Freezing CNN model")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 224 x 224
        # return:
        # - global_emb: [B, output_dim]
        # - local_emb: [B, output_dim, H, W] 如果 get_local=True
        model_name = self.cfg.model.vision.model_name.lower()
        if "resnet" in model_name or "resnext" in model_name:
            global_ft, local_ft_raw = self.resnet_forward(x)
        elif "densenet" in model_name:
            # 需要实现或抛错
            global_ft, local_ft_raw = self.densenet_forward(x)
        else:
            raise ValueError(f"Unsupported vision model: {self.cfg.model.vision.model_name}")
        # global_ft: [B, feature_dim]; local_ft_raw: [B, interm_feature_dim, H, W]

        # 投影
        global_emb = self.global_embedder(global_ft)  # [B, output_dim]
        local_emb = self.local_embedder(local_ft_raw)  # [B, output_dim, H, W]

        if get_local:
            return global_emb, local_emb
        else:
            return global_emb

    def resnet_forward(self, x):

        # --> fixed-size input: batch x 3 x 224 x 224
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        local_features = x # 提取attention map用
        x = self.model.layer4(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x, local_features

    def densenet_forward(self, x, extract_features=False):
        pass

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)


class PretrainedImageClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        num_cls: int,
        feature_dim: int,
        freeze_encoder: bool = True,
    ):
        super(PretrainedImageClassifier, self).__init__()
        self.img_encoder = image_encoder
        self.classifier = nn.Linear(feature_dim, num_cls)
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred


class ImageClassifier(nn.Module):
    def __init__(self, cfg, image_encoder=None):
        super(ImageClassifier, self).__init__()

        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.img_encoder, self.feature_dim, _ = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        self.classifier = nn.Linear(self.feature_dim, cfg.model.vision.num_targets)

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred
