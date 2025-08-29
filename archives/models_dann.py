import torch.nn as nn
import torch
from torch.autograd import Function, Variable
from layers import CNN, Encoder, HybirdDecoder

USE_CUDA = torch.cuda.is_available()

# 梯度反转层 (GRL - Gradient Reversal Layer)，它在前向传播时什么都不做，在反向传播时将梯度反转。
class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        return -lambda_ * grad_output, None

#领域判别器 (Domain Discriminator)，用来判断特征是来自源域（合成数据）还是目标域（真实数据）。
class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2) # 2个域: 0 for source, 1 for target
        )

    def forward(self, features):
        return self.discriminator(features)

# DNN模型类
class DANN_CNNSeq2Seq(nn.Module):
    def __init__(self, vocab_size, max_len, feature_dim=128, hidden_size=128):
        super(DANN_CNNSeq2Seq, self).__init__()
        self.max_len = max_len

        # --- 复用原有的模块 ---
        self.backbone = CNN()
        self.encoder = Encoder(rnn_hidden_size=feature_dim) # Encoder的输出维度即为特征维度
        self.decoder = HybirdDecoder(vocab_size=vocab_size, hidden_size=hidden_size)
        self.prediction = nn.Linear(hidden_size, vocab_size)

        # --- 新增的模块 ---
        self.grl = GradientReverseFunction.apply
        self.domain_discriminator = DomainDiscriminator(feature_dim=feature_dim)

        # --- 权重初始化---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, y=None, lambda_=1.0, mode='train'):
        """
        一个统一的 forward 函数
        - mode='train': 用于DANN训练，需要输入 x 和 y (仅用于源域)
        - mode='test': 用于评估，只需要输入 x
        """
        # 1. 提取共享特征
        cnn_out = self.backbone(x)
        encoder_outputs = self.encoder(cnn_out) # shape: (batch, seq_len, feature_dim)

        # 2. 分支一: 文本识别 (Label Predictor)
        label_predictions = None
        if mode == 'train':
            # 训练时使用 Teacher Forcing
            vocab_out = self.decoder.forward_train(encoder_outputs, self.max_len, y)
            label_predictions = self.prediction(vocab_out)
        elif mode == 'test':
            # 测试时逐个字符生成
            label_predictions = self.forward_test_step_by_step(x, encoder_outputs)

        # 3. 分支二: 领域分类 (Domain Discriminator)
        # 将序列特征平均池化，得到一个代表整张图的特征向量
        domain_features = torch.mean(encoder_outputs, dim=1) # (batch, feature_dim)
        # 应用 GRL
        reversed_features = self.grl(domain_features, lambda_)
        domain_predictions = self.domain_discriminator(reversed_features)

        return label_predictions, domain_predictions

    def forward_test_step_by_step(self, x, encoder_outputs):
        # 封装原来的 forward_test 逻辑
        outputs = []
        batch_size = x.size(0)
        input_char = torch.zeros([batch_size]).long()
        if USE_CUDA:
            input_char = input_char.cuda()

        last_hidden = Variable(torch.zeros(self.decoder.num_rnn_layers, batch_size, self.decoder.hidden_size))
        if USE_CUDA:
            last_hidden = last_hidden.cuda()

        for i in range(self.max_len - 1):
            output, last_hidden = self.decoder.forward_step(input_char, last_hidden, encoder_outputs)
            output = self.prediction(output)
            input_char = output.max(1)[1]
            outputs.append(output.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)
