import torch
import torch.nn as nn
import numpy as np
from torchvision.models.video import swin3d_t, Swin3D_T_Weights


class TimeSeriesTransformer(nn.Module):
    """
    Transformer for keypoint sequence encoding.
    """
    def __init__(self, input_dim, model_dim=256, num_heads=8, num_layers=4, output_dim=256, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.model_dim = model_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_fc = nn.Linear(model_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):  # x: (B, T, input_dim)
        B, T, _ = x.size()
        device = x.device

        # Sinusoidal positional encoding
        position = torch.arange(T, device=device).unsqueeze(0).unsqueeze(2)
        div_term = torch.exp(
            torch.arange(0, self.model_dim, 2, device=device) * -(np.log(10000.0) / self.model_dim)
        )
        pe = torch.zeros(1, T, self.model_dim, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        x = self.embedding(x) + pe
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_fc(x)
        x = self.norm(x)
        return x


class MultiModalActionModel(nn.Module):
    """
    Multimodal action recognition with fusion output only.
    """
    def __init__(self, num_classes=2, keypoint_dim=225):
        super(MultiModalActionModel, self).__init__()
        # Video backbone
        self.video_backbone = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
        self.video_backbone.head = nn.Identity()
        feature_dim_video = 768

        # Keypoint transformer
        self.keypoint_transformer = TimeSeriesTransformer(
            input_dim=keypoint_dim,
            model_dim=128,
            num_heads=4,
            num_layers=4,
            output_dim=128,
        )
        feature_dim_kp = 128

        # Fusion head
        fused_dim = feature_dim_video + feature_dim_kp
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        # Utility: separate heads for staged training if needed
        self.video_cls_head = nn.Sequential(
            nn.Linear(feature_dim_video, num_classes)
        )
        self.kp_cls_head = nn.Sequential(
            nn.Linear(feature_dim_kp, num_classes)
        )

    def forward(self, video, keypoint_seq, mode='fusion'):
        """
        mode: 'fusion', 'video', or 'keypoint'
        returns the specified prediction tensor.
        """
        video_feat = self.video_backbone(video)            # (B, 768)
        kp_feat = self.keypoint_transformer(keypoint_seq)   # (B, 128)

        if mode == 'video':
            return self.video_cls_head(video_feat)
        elif mode == 'keypoint':
            return self.kp_cls_head(kp_feat)
        # default: fusion
        fused = torch.cat([video_feat, kp_feat], dim=1)
        return self.fusion_head(fused)

    # Methods to freeze/unfreeze modules
    def freeze_video(self):
        for param in self.video_backbone.parameters(): param.requires_grad = False
    def unfreeze_video(self):
        for param in self.video_backbone.parameters(): param.requires_grad = True

    def freeze_keypoint(self):
        for param in self.keypoint_transformer.parameters(): param.requires_grad = False
    def unfreeze_keypoint(self):
        for param in self.keypoint_transformer.parameters(): param.requires_grad = True

    def freeze_fusion(self):
        for param in self.fusion_head.parameters(): param.requires_grad = False
    def unfreeze_fusion(self):
        for param in self.fusion_head.parameters(): param.requires_grad = True

    def freeze_video_cls(self):
        for param in self.video_cls_head.parameters(): param.requires_grad = False
    def unfreeze_video_cls(self):
        for param in self.video_cls_head.parameters(): param.requires_grad = True

    def freeze_kp_cls(self):
        for param in self.kp_cls_head.parameters(): param.requires_grad = False
    def unfreeze_kp_cls(self):
        for param in self.kp_cls_head.parameters(): param.requires_grad = True
