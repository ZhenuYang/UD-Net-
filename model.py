# ==========================================
# 文件名: model.py
# 功能: 定义 UD-Net/ResNet18 核心网络结构
# ==========================================
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes=64):
        super().__init__()
        # 1. 加载官方预训练模型 (ImageNet权重)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. 冻结前面层的参数 (部分冻结策略，防止小样本过拟合)
        for name, param in self.backbone.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        
        # 获取预训练模型全连接层的输入维度 (ResNet18是512)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 3. 定义新的双头输出
        self.dropout = nn.Dropout(p=0.5)
        
        # 分支A: 预测均值 (Logits)
        self.fc_mean = nn.Linear(in_features, num_classes)
        
        # 分支B: 预测方差 (Sigma)
        self.fc_sigma = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Softplus()
        )
        
        # 初始化 Sigma 分支
        if isinstance(self.fc_sigma[0], nn.Linear):
            nn.init.constant_(self.fc_sigma[0].bias, 3.0) 
            nn.init.normal_(self.fc_sigma[0].weight, std=0.001)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        
        logits = self.fc_mean(features)
        sigma = self.fc_sigma(features)
        sigma = torch.clamp(sigma, min=3.0, max=15.0)
        
        return logits, sigma

def ResNet18():
    return ResNet(num_classes=64)