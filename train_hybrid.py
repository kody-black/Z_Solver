import argparse
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
import matplotlib
from torch.autograd import Variable
import pprint
import numpy as np
import random
import time
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models_hybrid import Stable_Hybrid_Model
from datasets import load_datasets_mean_teacher
from util import compute_seq_acc, Seq2SeqLoss, ConsistentLoss, get_current_consistency_weight

parser = argparse.ArgumentParser(description='PyTorch Captcha Training with Delayed Attention-DANN')
# --- 数据与模型参数 ---
parser.add_argument('--dataset', default='weibo', type=str, help="数据集名称")
parser.add_argument('--label', default="10000.txt", type=str, help='带标签文件名')
parser.add_argument('--batch-size', default=32, type=int, help='带标签数据(源域)的批次大小')
parser.add_argument('--secondary-batch-size', default=64, type=int, help='无标签数据(目标域)的批次大小')
parser.add_argument('--unlabeled-number', default=5000, type=int, help='使用的无标签图片数量')

# --- 训练超参数 ---
parser.add_argument('--epoch', default=400, type=int, help='训练轮数')
parser.add_argument('--lr', default=1e-3, type=float, help='学习率 (AdamW 的稳定值)')
parser.add_argument('--clip-grad', default=5.0, type=float, help='梯度裁剪的范数上限')
parser.add_argument('--seed', default=42, type=int, help='随机种子')

# --- 损失权重与调度参数 ---
parser.add_argument('--warmup-epochs', default=10, type=int, help='只进行监督学习的热身轮数')
parser.add_argument('--attn-warmup-epochs', default=50, type=int, help='注意力DANN介入的热身轮数 (在此之前只用全局池化)')
parser.add_argument('--threshold', default=0.95, type=float, help='伪标签置信度阈值')
parser.add_argument('--weight', default=1.0, type=float, help='一致性损失权重')
parser.add_argument('--consistency-rampup', default=50, type=int, help='一致性损失权重斜坡上升的周期')
parser.add_argument('--lambda-d', default=0.1, type=float, help='领域对抗损失的权重')
args = parser.parse_args()

# --- 环境设置 ---
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True
pprint.pprint(args)
USE_CUDA = torch.cuda.is_available()
if not os.path.exists('result'): os.makedirs('result')

# --- 数据加载 ---
source_loader, target_loader, test_loader, id2token, MAXLEN, _ = load_datasets_mean_teacher(args)
print("Vocabulary:", "".join(list(id2token.values())))

# --- 模型、损失函数、优化器 ---
model = Stable_Hybrid_Model(vocab_size=len(id2token), max_len=MAXLEN, feature_dim=128)
model_ema = Stable_Hybrid_Model(vocab_size=len(id2token), max_len=MAXLEN, feature_dim=128)

class_criterion = Seq2SeqLoss()
consistent_criterion = ConsistentLoss(args.threshold)
domain_criterion = CrossEntropyLoss()

if USE_CUDA:
    model, model_ema = model.cuda(), model_ema.cuda()
    class_criterion, consistent_criterion, domain_criterion = class_criterion.cuda(), consistent_criterion.cuda(), domain_criterion.cuda()

for param_main, param_ema in zip(model.parameters(), model_ema.parameters()):
    param_ema.data.copy_(param_main.data)
    param_ema.requires_grad = False

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

# --- 训练历史记录 ---
history = {'train_loss_class': [], 'train_loss_consistency': [], 'train_loss_domain': [], 'train_accuracy': [], 'test_accuracy': [], 'test_accuracy_ema': []}
best_ema_acc = 0.0

# --- 打印表头 ---
TABLE_HEADER = "| Epoch | L_Class | L_Cons  | L_Domain| Train Acc | Test Acc | Test Acc EMA | Time (s) |"
TABLE_SEPARATOR = "-" * len(TABLE_HEADER)
print(TABLE_SEPARATOR)
print(TABLE_HEADER)
print(TABLE_SEPARATOR)

# --- 训练循环 ---
for epoch in range(args.epoch):
    time_epoch = time.time()
    model.train(); model_ema.train()

    running_losses = {'class': 0.0, 'cons': 0.0, 'domain': 0.0}
    running_accuracy = 0.0
    
    len_dataloader = len(target_loader)
    source_iter = iter(source_loader)

    for i, (inputs_u_w, inputs_u_s) in enumerate(target_loader):
        try:
            inputs_x, targets_x = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            inputs_x, targets_x = next(source_iter)

        if USE_CUDA:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
            inputs_u_w, inputs_u_s = inputs_u_w.cuda(), inputs_u_s.cuda()

        optimizer.zero_grad()
        
        # 1. 监督损失 (始终计算以获取特征)
        logits_l, source_features, source_weight_matrix = model.forward_supervised(inputs_x, targets_x)
        Lx = class_criterion(logits_l, targets_x)
        
        # 根据是否在热身期决定损失构成
        if epoch < args.warmup_epochs:
            loss_all = Lx
        else:
            # 2. 一致性损失
            with torch.no_grad():
                logits_u_w_teacher, _, _ = model_ema.forward_unsupervised(inputs_u_w)
            
            logits_u_s, target_features, target_weight_matrix = model.forward_unsupervised(inputs_u_s)
            
            consistency_weight = get_current_consistency_weight(args.weight, args.consistency_rampup, epoch - args.warmup_epochs)
            Lu = consistency_weight * consistent_criterion(logits_u_w_teacher.detach(), logits_u_s)

            # 3. 领域对抗损失
            p = float(i + (epoch - args.warmup_epochs) * len_dataloader) / ((args.epoch - args.warmup_epochs) * len_dataloader)
            lambda_grl = 2. / (1. + np.exp(-10 * p)) - 1
            
            # --- 改动: 注意力延迟介入 ---
            use_attention_for_dann = epoch >= args.attn_warmup_epochs

            domain_preds_s, domain_preds_t = model.forward_domain(
                source_features, target_features, 
                source_weight_matrix.detach(), target_weight_matrix.detach(), 
                lambda_grl, use_attn=use_attention_for_dann
            )

            labels_s = torch.zeros(domain_preds_s.size(0), dtype=torch.long, device=inputs_x.device)
            labels_t = torch.ones(domain_preds_t.size(0), dtype=torch.long, device=inputs_x.device)
            
            Ld = domain_criterion(torch.cat((domain_preds_s, domain_preds_t)), torch.cat((labels_s, labels_t)))

            # 4. 总损失
            loss_all = Lx + Lu + args.lambda_d * Ld
            running_losses['cons'] += Lu.item()
            running_losses['domain'] += Ld.item()

        if torch.isnan(loss_all):
            print(f"NaN detected in total loss at epoch {epoch}, iteration {i}. Stopping.")
            exit()

        loss_all.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        optimizer.step()

        if epoch >= args.warmup_epochs:
            for ema_param, param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.mul_(0.999).add_(0.001, param.data)

        with torch.no_grad():
            _, acc = compute_seq_acc(logits_l, targets_x, MAXLEN)
        running_accuracy += acc
        running_losses['class'] += Lx.item()
    
    # --- 保存和打印统计信息 ---
    history['train_loss_class'].append(running_losses['class'] / len_dataloader)
    history['train_loss_consistency'].append(running_losses['cons'] / len_dataloader)
    history['train_loss_domain'].append(running_losses['domain'] / len_dataloader)
    history['train_accuracy'].append(running_accuracy / (len_dataloader * args.batch_size))

    # --- 评估 ---
    model.eval(); model_ema.eval()
    test_accuracy, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            if USE_CUDA: x, y = x.cuda(), y.cuda()
            outputs, _, _ = model.forward_unsupervised(x)
            _, acc = compute_seq_acc(outputs, y, MAXLEN)
            test_accuracy += acc
            total += y.size(0)
    history['test_accuracy'].append(test_accuracy / total if total > 0 else 0)

    test_accuracy_ema, total_ema = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            if USE_CUDA: x, y = x.cuda(), y.cuda()
            outputs_ema, _, _ = model_ema.forward_unsupervised(x)
            _, acc_ema = compute_seq_acc(outputs_ema, y, MAXLEN)
            test_accuracy_ema += acc_ema
            total_ema += y.size(0)
    current_ema_acc = test_accuracy_ema / total_ema if total_ema > 0 else 0
    history['test_accuracy_ema'].append(current_ema_acc)

    epoch_time = time.time() - time_epoch
    print(f"| {epoch:^5} | {history['train_loss_class'][-1]:<7.4f} | {history['train_loss_consistency'][-1]:<7.4f} | {history['train_loss_domain'][-1]:<7.4f} | {history['train_accuracy'][-1]:<9.4f} | {history['test_accuracy'][-1]:<8.4f} | {history['test_accuracy_ema'][-1]:<12.4f} | {epoch_time:<8.2f} |")

    if current_ema_acc > best_ema_acc:
        best_ema_acc = current_ema_acc
        torch.save(model_ema.state_dict(), f"result/best_model_ema.pth")
        
print(TABLE_SEPARATOR)
print(f"Training finished. Best EMA model accuracy: {best_ema_acc:.4f}")

# --- 绘图 ---
fig = plt.figure(figsize=(20, 8))
fig.suptitle(f'Training History for {args.dataset} - Delayed Attention DANN')
ax1 = fig.add_subplot(121)
ax1.set_title("Loss Curves")
ax1.plot(history['train_loss_class'], label='L_Class (Train)')
ax1.plot(history['train_loss_consistency'], label='L_Cons (Train)')
ax1.plot(history['train_loss_domain'], label='L_Domain (Train)')
ax1.legend()
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(bottom=0)
ax2 = fig.add_subplot(122)
ax2.set_title("Accuracy Curves")
ax2.plot(history['train_accuracy'], label='Train Accuracy')
ax2.plot(history['test_accuracy'], label='Test Acc (Student)')
ax2.plot(history['test_accuracy_ema'], label='Test Acc (EMA/Teacher)')
ax2.axhline(y=best_ema_acc, color='g', linestyle='--', label=f'Best EMA Acc: {best_ema_acc:.4f}')
ax2.axvline(x=args.attn_warmup_epochs, color='r', linestyle=':', label=f'Attn DANN Start (Epoch {args.attn_warmup_epochs})')
ax2.legend()
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_ylim(0, 1.0)
path_params = f"DelayedAttnDANN_{args.dataset}_{args.label.split('.')[0]}_{args.attn_warmup_epochs}"
fig.savefig(f"result/{path_params}.png")
