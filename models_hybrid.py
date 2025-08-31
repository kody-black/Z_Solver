import torch.nn as nn
import torch
from torch.autograd import Function, Variable
from layers import CNN, Encoder, HybirdDecoder

USE_CUDA = torch.cuda.is_available()

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        return -lambda_ * grad_output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, features):
        return self.discriminator(features)

class Stable_Hybrid_Model(nn.Module):
    def __init__(self, vocab_size, max_len, feature_dim=128, hidden_size=128):
        super(Stable_Hybrid_Model, self).__init__()
        self.max_len = max_len

        self.backbone = CNN()
        self.encoder = Encoder(rnn_hidden_size=feature_dim)
        self.decoder = HybirdDecoder(vocab_size=vocab_size, hidden_size=hidden_size)
        self.prediction = nn.Linear(hidden_size, vocab_size)

        self.grl_func = GradientReverseFunction.apply
        self.domain_discriminator = DomainDiscriminator(feature_dim=feature_dim)

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
        cnn_out = self.backbone(x)
        encoder_outputs = self.encoder(cnn_out)
        return encoder_outputs

    def forward_supervised(self, x, y):
        encoder_outputs = self.extract_features(x)
        decoder_out, weight_matrix = self.decoder.forward_train(encoder_outputs, self.max_len, y)
        logits = self.prediction(decoder_out)
        return logits, encoder_outputs, weight_matrix

    def forward_unsupervised(self, x):
        encoder_outputs = self.extract_features(x)
        
        outputs = []
        batch_size = x.size(0)
        # 确保 input_char 和 last_hidden 在正确的设备上
        device = x.device
        input_char = torch.zeros([batch_size], dtype=torch.long, device=device)
        last_hidden = torch.zeros(self.decoder.num_rnn_layers, batch_size, self.decoder.hidden_size, device=device)

        weight_matrices = []
        for i in range(self.max_len - 1):
            feature, last_hidden, weight_matrix = self.decoder.forward_step(input_char, last_hidden, encoder_outputs)
            output = self.prediction(feature)
            input_char = output.max(1)[1]
            outputs.append(output.unsqueeze(1))
            weight_matrices.append(weight_matrix)
        
        return torch.cat(outputs, dim=1), encoder_outputs, torch.cat(weight_matrices, dim=1)

    def forward_domain(self, source_features, target_features, source_weights, target_weights, lambda_, use_attn=False):
        """
        路径3: 领域判别路径
        - use_attn: bool, 控制是否使用注意力加权特征
        """
        # 1. 全局池化特征 (始终使用)
        source_pooled = torch.mean(source_features, dim=1)
        target_pooled = torch.mean(target_features, dim=1)
        
        # 2. 根据 use_attn 标志决定是否使用注意力加权特征
        if use_attn:
            source_weighted = torch.bmm(source_weights, source_features)
            source_attn = source_weighted.view(-1, source_weighted.shape[-1])
            
            target_weighted = torch.bmm(target_weights, target_features)
            target_attn = target_weighted.view(-1, target_weighted.shape[-1])

            combined_source = torch.cat([source_pooled, source_attn], dim=0)
            combined_target = torch.cat([target_pooled, target_attn], dim=0)
        else:
            combined_source = source_pooled
            combined_target = target_pooled

        # 3. 应用 GRL 并进行判别
        reversed_source = self.grl_func(combined_source, lambda_)
        reversed_target = self.grl_func(combined_target, lambda_)

        preds_s = self.domain_discriminator(reversed_source)
        preds_t = self.domain_discriminator(reversed_target)
        
        return preds_s, preds_t
