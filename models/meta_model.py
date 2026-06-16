import torch.nn as nn
from transformers import BertModel
from models.autoencoder import AutoEncoder
from utils.maml import MAML
import torch
import torch.nn.functional as F
class MultiModalModel(nn.Module):
    """多模态情感分类模型（单标签分类）"""

    def __init__(self, n_labels, model_path, feature_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.cls_temperature = nn.Parameter(torch.tensor(0.5))  # 分类损失温度
        self.recon_temperature = nn.Parameter(torch.tensor(0.5))  # 重建损失温度

        # 文本模态处理
        self.text_encoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 情感模态处理
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fusion_gate = nn.Linear(512, 256)
        nn.init.xavier_uniform_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)
        # 自编码器模块
        self.autoencoder = AutoEncoder(input_dim=512, latent_dim=64,lambda_gp=10 )

        # 分类器
        # 原型分类器
        # self.classifier = PrototypeClassifier(feat_dim=64, n_labels=n_labels)

        self.classifier = nn.Linear(64, n_labels)
        # 冻结BERT的大部分层，只微调后几层
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, bert_features, labels=None):
        # 文本模态处理
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_encoder(text_output.last_hidden_state[:, 0, :])

        # 情感模态处理
        sentiment_features = self.sentiment_encoder(bert_features)

        # ===== Channel-weighted fusion (gating) =====
        fuse_in = torch.cat([text_features, sentiment_features], dim=1)  # [B, 512]
        g = torch.sigmoid(self.fusion_gate(fuse_in))  # [B, 256]

        text_fused = g * text_features  # [B, 256]
        senti_fused = (1.0 - g) * sentiment_features  # [B, 256]

        combined = torch.cat([text_fused, senti_fused], dim=1)  # [B, 512]

        # 自编码器处理
        latent, reconstructed = self.autoencoder(combined)

        # 分类
        logits = self.classifier(latent)


        loss = None
        if labels is not None:
            # 分类损失
            cls_temp = torch.clamp(self.cls_temperature, 0.1, 10.0)
            recon_temp = torch.clamp(self.recon_temperature, 0.1, 10.0)

            # 分类损失（温度缩放）
            # scaled_cls_loss = F.cross_entropy(logits / cls_temp, labels)

            # 重建损失（温度调权）
            # scaled_recon_loss = nn.MSELoss()(reconstructed, combined) / recon_temp
            scaled_cls_loss = F.cross_entropy(logits, labels)#消融part
            scaled_recon_loss = F.mse_loss(reconstructed, combined)#消融part
            # 总损失
            loss = scaled_cls_loss + 1   * scaled_recon_loss

        return loss, logits


class PrototypeClassifier(nn.Module):
    def __init__(self, feat_dim=512, n_labels=30):
        super().__init__()
        # 可学习的类原型向量（每个标签对应一个原型）
        self.prototypes = nn.Parameter(torch.randn(n_labels, feat_dim))  # [n_labels, feat_dim]
        # 原型向量归一化层（提升距离度量的稳定性）
        self.norm = nn.LayerNorm(feat_dim, elementwise_affine=False)  # 无参数归一化[8](@ref)

    def forward(self, features, labels=None):
        """
        Args:
            features: 输入特征 [batch_size, feat_dim]
            labels: 仅用于训练时计算对比损失（可选）
        Returns:
            logits: 负欧氏距离 [batch_size, n_labels]
        """
        # 归一化特征和原型（避免尺度敏感）
        features_norm = self.norm(features)  # [batch, feat_dim]
        prototypes_norm = self.norm(self.prototypes)  # [n_labels, feat_dim]

        # 计算特征与所有原型的欧氏距离
        dist = torch.cdist(features_norm, prototypes_norm, p=2)  # [batch, n_labels]

        # 返回负距离作为logits（距离越小，概率越大）
        return -dist