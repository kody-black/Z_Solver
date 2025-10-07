import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import subprocess
import json


def sigmoid_rampup(current, rampup_length):
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))
        

def get_current_consistency_weight(consistency, consistency_rampup, epoch):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)
    

class Seq2SeqLoss(nn.Module):
    def __init__(self):
        super(Seq2SeqLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, y):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        y: [batch_size, max_len]
        """
        max_len = y.size(1)

        loss = sum([self.criterion(outputs[:, i, :], y[:, i + 1]) for i in range(max_len - 1)]) / (max_len - 1)

        return loss


class ConsistentLoss_MT(nn.Module):
    def __init__(self):
        super(ConsistentLoss_MT, self).__init__()

    def forward(self, outputs, outputs_ema):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        outputs_ema: [batch_size, max_len-1, vocab_size]
        """
        max_len = outputs.size(1) + 1

        loss = 0
        for i in range(max_len-1):
            input_logits = outputs[:, i, :]
            target_logits = outputs_ema[:, i, :]
            input_softmax = F.softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)
            loss += F.mse_loss(input_softmax, target_softmax, reduction='none').mean()

        return loss / (max_len - 1)


class ConsistentLoss_MT_Temperature(nn.Module):
    def __init__(self, threshold):
        super(ConsistentLoss_MT_Temperature, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, outputs_ema):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        outputs_ema: [batch_size, max_len-1, vocab_size]
        """
        max_len = outputs.size(1) + 1

        loss = 0
        for i in range(max_len-1):
            input_logits = outputs[:, i, :]
            target_logits = outputs_ema[:, i, :]
            input_softmax = F.softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)
            max_probs, targets_u = torch.max(target_softmax, dim=-1)
            mask = max_probs.ge(self.threshold).float()

            loss += (F.mse_loss(input_softmax, target_softmax, reduction='none').mean(dim=-1) * mask).mean()

        return loss / (max_len - 1)

class ConsistentLoss(nn.Module):
    def __init__(self, threshold):
        super(ConsistentLoss, self).__init__()

        self.threshold = threshold

    def compute_consistent_loss(self, logits_u_w, logits_u_s):
        """
        outputs: [secondary_batch_size, max_len-1, vocab_size]
        outputs_ema: [secondary_batch_size, max_len-1, vocab_size]
        """
        max_len = logits_u_s.size(1) + 1

        pseudo_label = torch.softmax(logits_u_w, dim=-1)
        loss_all = 0
        for i in range(max_len - 1):
            max_probs, targets_u = torch.max(pseudo_label[:, i, :], dim=-1)
            mask = max_probs.ge(self.threshold).float()
            loss_all += (F.cross_entropy(logits_u_s[:, i, :], targets_u,
                                         reduction='none') * mask).mean()

        return loss_all / (max_len - 1)

    def forward(self, logits_u_w, logits_u_s):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        outputs_ema: [batch_size, max_len-1, vocab_size]
        """
        loss = self.compute_consistent_loss(logits_u_w, logits_u_s)

        return loss


def compute_seq_acc(outputs, y, max_len):
    """
    outputs: [batch_size, max_len-1, vocab_size]
    y: [batch_size, max_len]
    """

    accuracy_clevel, accuracy_all = compute_acc_step(outputs, y, max_len)

    return accuracy_clevel, accuracy_all


def compute_acc_step(outputs, y, max_len):
    num_eq = (y[:, 1:].data == outputs.max(2)[1]).sum(dim=1)
    accuracy_clevel = num_eq.sum() / (max_len - 1)
    accuracy_all = (num_eq == max_len - 1).sum()

    return accuracy_clevel.item(), accuracy_all.item()


def get_gpu_memory_usage():
    """获取所有GPU的内存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id = int(parts[0])
                    memory_used = int(parts[1])
                    memory_total = int(parts[2])
                    gpu_util = int(parts[3])
                    gpu_info.append({
                        'id': gpu_id,
                        'memory_used': memory_used,
                        'memory_total': memory_total,
                        'memory_free': memory_total - memory_used,
                        'utilization': gpu_util
                    })
        return gpu_info
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return []


def select_best_gpu(min_memory_mb=1000, max_utilization=50):
    """
    选择最空闲的GPU
    
    Args:
        min_memory_mb: 最小可用内存要求（MB）
        max_utilization: 最大GPU利用率阈值（%）
    
    Returns:
        int: 选定的GPU ID，如果没有合适的GPU则返回0
    """
    if not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU")
        return None
    
    gpu_info = get_gpu_memory_usage()
    if not gpu_info:
        print("无法获取GPU信息，使用默认GPU 0")
        return 0
    
    # 过滤符合条件的GPU
    available_gpus = []
    for gpu in gpu_info:
        if gpu['memory_free'] >= min_memory_mb and gpu['utilization'] <= max_utilization:
            available_gpus.append(gpu)
    
    if not available_gpus:
        print(f"没有找到符合条件的GPU（内存>={min_memory_mb}MB，利用率<={max_utilization}%）")
        print("可用GPU信息:")
        for gpu in gpu_info:
            print(f"  GPU {gpu['id']}: 内存 {gpu['memory_free']}/{gpu['memory_total']}MB, 利用率 {gpu['utilization']}%")
        print("使用GPU 0")
        return 0
    
    # 按内存空闲量和利用率排序，选择最优的
    available_gpus.sort(key=lambda x: (x['memory_free'], -x['utilization']), reverse=True)
    best_gpu = available_gpus[0]
    
    print(f"选择GPU {best_gpu['id']}: 内存 {best_gpu['memory_free']}/{best_gpu['memory_total']}MB, 利用率 {best_gpu['utilization']}%")
    return best_gpu['id']


def setup_gpu(device_id=None, min_memory_mb=1000, max_utilization=50):
    """
    设置GPU设备
    
    Args:
        device_id: 指定的GPU ID，如果为None则自动选择
        min_memory_mb: 最小可用内存要求（MB）
        max_utilization: 最大GPU利用率阈值（%）
    
    Returns:
        torch.device: 选定的设备
    """
    if not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        return torch.device('cpu')
    
    if device_id is None:
        device_id = select_best_gpu(min_memory_mb, max_utilization)
    
    if device_id is None:
        return torch.device('cpu')
    
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)
    print(f"使用设备: {device}")
    return device

