import torch.nn as nn
import torch
from torch.autograd import Function, Variable
from layers import CNN, Encoder, HybirdDecoder

USE_CUDA = torch.cuda.is_available()

# =========================================================================================
# 1. GRL (Gradient Reversal Layer) - DANN的核心
# =========================================================================================
class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        return -lambda_ * grad_output, None

# =========================================================================================
# 2. 领域判别器 (Domain Discriminator)
# =========================================================================================
class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, features):
        return self.discriminator(features)

# =========================================================================================
# 3. 全新的、更稳定的混合模型
# =========================================================================================
class Stable_Hybrid_Model(nn.Module):
    def __init__(self, vocab_size, max_len, feature_dim=128, hidden_size=128):
        super(Stable_Hybrid_Model, self).__init__()
        self.max_len = max_len

        # --- 基础模块 (与之前相同) ---
        self.backbone = CNN()
        self.encoder = Encoder(rnn_hidden_size=feature_dim)
        self.decoder = HybirdDecoder(vocab_size=vocab_size, hidden_size=hidden_size)
        self.prediction = nn.Linear(hidden_size, vocab_size)

        # --- DANN 模块 ---
        self.grl_func = GradientReverseFunction.apply
        self.domain_discriminator = DomainDiscriminator(feature_dim=feature_dim)

        # --- 权重初始化 ---
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def extract_features(self, x):
        """辅助函数：只提取特征"""
        cnn_out = self.backbone(x)
        encoder_outputs = self.encoder(cnn_out)
        return encoder_outputs

    def forward_supervised(self, x, y):
        """
        路径1: 监督学习路径 (用于计算 L_Class)
        使用 Teacher Forcing，返回 logits。
        """
        encoder_outputs = self.extract_features(x)
        decoder_out = self.decoder.forward_train(encoder_outputs, self.max_len, y)
        logits = self.prediction(decoder_out)
        return logits

    def forward_unsupervised(self, x):
        """
        路径2: 无监督/测试路径 (用于计算 L_Cons 和最终评估)
        使用自回归解码，返回 logits。
        """
        encoder_outputs = self.extract_features(x)
        
        outputs = []
        batch_size = x.size(0)
        input_char = torch.zeros([batch_size]).long().cuda() if USE_CUDA else torch.zeros([batch_size]).long()
        last_hidden = Variable(torch.zeros(self.decoder.num_rnn_layers, batch_size, self.decoder.hidden_size)).cuda() if USE_CUDA else Variable(torch.zeros(self.decoder.num_rnn_layers, batch_size, self.decoder.hidden_size))

        for i in range(self.max_len - 1):
            output, last_hidden = self.decoder.forward_step(input_char, last_hidden, encoder_outputs)
            output = self.prediction(output)
            input_char = output.max(1)[1]
            outputs.append(output.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)

    def forward_domain(self, features, lambda_):
        """
        路径3: 领域判别路径 (用于计算 L_Domain)
        接收提取好的特征，返回领域预测。
        """
        # 将序列特征平均池化
        pooled_features = torch.mean(features, dim=1)
        # 应用 GRL
        reversed_features = self.grl_func(pooled_features, lambda_)
        domain_predictions = self.domain_discriminator(reversed_features)
        return domain_predictions
