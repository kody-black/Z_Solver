import argparse
import torch
from torch import optim
from torch.nn import CrossEntropyLoss # 引入领域分类损失
import matplotlib
from torch.autograd import Variable
import pprint
import numpy as np
import random
import time

# 确保 matplotlib 后端正确设置
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models_dann import DANN_CNNSeq2Seq
from datasets import load_datasets_mean_teacher
from util import compute_seq_acc, Seq2SeqLoss

parser = argparse.ArgumentParser(description='PyTorch Captcha Training Using DANN')

parser.add_argument('--dataset', default='yandex', type=str, help="数据集名称 (例如 'google')")
parser.add_argument('--label', default="10000.txt", type=str, help='用于训练的带标签文件名')
parser.add_argument('--batch-size', default=32, type=int, help='带标签数据(源域)的批次大小')
parser.add_argument('--secondary-batch-size', default=32, type=int, help='无标签数据(目标域)的批次大小')
parser.add_argument('--unlabeled-number', default=5000, type=int, help='使用的无标签图片数量')
parser.add_argument('--epoch', default=400, type=int, help='训练轮数')
parser.add_argument('--lr', default=0.02, type=float, help='学习率')
parser.add_argument('--seed', default=42, type=int, help='随机种子')
parser.add_argument('--lambda-weight', default=1.0, type=float, help='领域对抗损失的权重')

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

LR = args.lr
NUM_EPOCHS = args.epoch

# 复用 load_datasets_mean_teacher 
# dataloader_train_labeled -> source_loader (合成数据)
# dataloader_train_nolabeled -> target_loader (真实数据, 来自 'buchong' 文件夹)
source_loader, target_loader, test_loader, id2token, MAXLEN, _ = load_datasets_mean_teacher(args)
print("Vocabulary:", "".join(list(id2token.values())))
print(f"Source (labeled) batches: {len(source_loader)}, Target (unlabeled) batches: {len(target_loader)}")


# 1. 实例化 DANN 模型
# Encoder 输出维度是 128，所以 feature_dim=128
model = DANN_CNNSeq2Seq(vocab_size=len(id2token), max_len=MAXLEN, feature_dim=128)

# 2. 定义损失函数
label_criterion = Seq2SeqLoss()         # 文本识别损失
domain_criterion = CrossEntropyLoss()   # 领域分类损失

if USE_CUDA:
    model = model.cuda()
    label_criterion = label_criterion.cuda()
    domain_criterion = domain_criterion.cuda()

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)

history = {
    'train_loss_label': [], 'train_loss_domain': [], 'train_accuracy': [],
    'test_loss': [], 'test_accuracy': []
}

# --- 训练循环 ---
print("Starting DANN training...")
for epoch in range(NUM_EPOCHS):
    time_epoch = time.time()
    model.train()

    # 确保两个 dataloader 长度一致进行迭代
    len_dataloader = min(len(source_loader), len(target_loader))
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    
    running_loss_label, running_loss_domain, running_accuracy, running_accclevel = 0.0, 0.0, 0.0, 0.0

    for i in range(len_dataloader):
        source_inputs, source_labels = next(source_iter)
        # 目标域数据来自原来的 'nolabeled' 加载器，它返回两个增强版本，只用第一个
        target_inputs, _ = next(target_iter)

        if USE_CUDA:
            source_inputs, source_labels = source_inputs.cuda(), source_labels.cuda()
            target_inputs = target_inputs.cuda()

        optimizer.zero_grad()

        # 动态调整 GRL 的 lambda 参数
        p = float(i + epoch * len_dataloader) / (NUM_EPOCHS * len_dataloader)
        lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
        
        # 2. 前向传播
        # --- 处理源域 (有标签) ---
        label_preds_s, domain_preds_s = model(source_inputs, y=source_labels, lambda_=lambda_, mode='train')
        
        # --- 处理目标域 (无标签) ---
        # 目标域没有 y，所以不计算 label loss，只获取 domain prediction
        # mode='test' 在这里只是为了避免计算 decoder 部分，节省计算资源
        _, domain_preds_t = model(target_inputs, y=None, lambda_=lambda_, mode='test') 
        
        # 3. 计算损失
        # 3.1 文本识别损失 (只在源域上)
        loss_label = label_criterion(label_preds_s, source_labels)

        # 3.2 领域分类损失 (在源域和目标域上)
        domain_preds = torch.cat((domain_preds_s, domain_preds_t), dim=0)
        domain_labels_source = torch.zeros(source_inputs.size(0)).long().cuda()
        domain_labels_target = torch.ones(target_inputs.size(0)).long().cuda()
        domain_labels = torch.cat((domain_labels_source, domain_labels_target), dim=0)
        
        loss_domain = domain_criterion(domain_preds, domain_labels)

        # 3.3 总损失
        total_loss = loss_label + args.lambda_weight * loss_domain

        # 4. 反向传播和优化
        total_loss.backward()
        optimizer.step()

        # 记录统计数据
        running_loss_label += loss_label.item()
        running_loss_domain += loss_domain.item()
        
        # 计算源域上的训练精度
        acccl, acc = compute_seq_acc(label_preds_s, source_labels, MAXLEN)
        running_accclevel += acccl / source_inputs.size(0)
        running_accuracy += acc / source_inputs.size(0)

    # --- 训练轮次结束，打印统计信息 ---
    history['train_loss_label'].append(running_loss_label / len_dataloader)
    history['train_loss_domain'].append(running_loss_domain / len_dataloader)
    history['train_accuracy'].append(running_accuracy / len_dataloader)
    print(f"Epoch: {epoch}/{NUM_EPOCHS} | Time: {time.time() - time_epoch:.2f}s")
    print(f"\tTrain Label Loss: {history['train_loss_label'][-1]:.4f} | "
          f"Train Domain Loss: {history['train_loss_domain'][-1]:.4f} | "
          f"Train Accuracy: {history['train_accuracy'][-1]:.4f}")

    # --- 评估循环 ---
    model.eval()
    test_loss, test_accuracy, test_accclevel, total = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            if USE_CUDA:
                x, y = x.cuda(), y.cuda()
            
            # 评估时，只关心文本识别结果
            outputs, _ = model(x, mode='test')
            
            loss_batch = label_criterion(outputs, y)
            acccl, acc = compute_seq_acc(outputs, y, MAXLEN)

            test_loss += loss_batch.item()
            test_accclevel += acccl
            test_accuracy += acc
            total += y.size(0)

    history['test_loss'].append(test_loss / len(test_loader))
    history['test_accuracy'].append(test_accuracy / total)
    print(f"\tTest Loss: {history['test_loss'][-1]:.4f} | "
          f"Test Accuracy: {history['test_accuracy'][-1]:.4f} ({test_accuracy}/{total})")

# --- 训练结束，绘制结果图 ---
fig = plt.figure(figsize=(20, 8))
# 损失曲线
ax1 = fig.add_subplot(121)
ax1.set_title("Loss Curves")
ax1.plot(history['train_loss_label'], 'r', label='Train Label Loss')
ax1.plot(history['train_loss_domain'], 'y', label='Train Domain Loss')
ax1.plot(history['test_loss'], 'b', label='Test Loss')
ax1.legend()
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")

# 准确率曲线
ax2 = fig.add_subplot(122)
ax2.set_title("Accuracy Curves")
ax2.plot(history['train_accuracy'], 'r', label='Train Accuracy')
ax2.plot(history['test_accuracy'], 'b', label='Test Accuracy')
# 标注最高测试准确率
max_acc = max(history['test_accuracy'])
max_epoch = history['test_accuracy'].index(max_acc)
ax2.annotate(f"Max: {max_acc:.4f}", xy=(max_epoch, max_acc), xytext=(max_epoch, max_acc*0.9),
             arrowprops=dict(facecolor='black', shrink=0.05))
ax2.legend()
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")

# 保存结果图
path = f"DANN_{args.dataset}_{args.label.split('.')[0]}_{args.unlabeled_number}_{args.lr}_{args.seed}"
fig.savefig(f"result/{path}.png")
print(f"Result plot saved to result/{path}.png")
