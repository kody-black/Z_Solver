import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import os

def parse_log_file(log_file_path, debug=False):
    """
    解析训练日志文件，提取训练数据
    
    Args:
        log_file_path: 日志文件路径
        debug: 是否显示调试信息
        
    Returns:
        dict: 包含所有训练数据的字典
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"日志文件不存在: {log_file_path}")
    
    # 初始化数据存储
    data = {
        'epochs': [],
        'train_loss_class': [],
        'train_loss_consistency': [],
        'train_loss_domain': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'test_accuracy_ema': [],
        'time': []
    }
    
    # 解析配置参数
    config = {}
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 提取配置信息
    for line in lines:
        if line.startswith('Namespace('):
            # 解析配置参数
            config_str = line.strip()
            # 提取dataset名称
            dataset_match = re.search(r"dataset='(\w+)'", config_str)
            if dataset_match:
                config['dataset'] = dataset_match.group(1)
            
            # 提取其他参数
            param_patterns = {
                'batch_size': r'batch_size=(\d+)',
                'unlabeled_number': r'unlabeled_number=(\d+)',
                'lr': r'lr=([\d.e-]+)',
                'seed': r'seed=(\d+)',
                'epoch': r'epoch=(\d+)'
            }
            
            for param, pattern in param_patterns.items():
                match = re.search(pattern, config_str)
                if match:
                    config[param] = match.group(1)
            break
    
    # 解析训练数据
    data_lines_found = 0
    for i, line in enumerate(lines):
        # 匹配数据行格式：|   0   | 2.8976  | 0.0000  | 0.6004  | 0.0000    | 0.0000   | 0.0000       | 69.42    |
        if line.startswith('|') and '|' in line and not line.strip().startswith('| Epoch') and not line.strip().startswith('|---'):
            if debug:
                print(f"Line {i+1}: {line.strip()}")
            
            # 去掉开头和结尾的 '|'，然后按 '|' 分割
            line_clean = line.strip().strip('|')
            parts = [part.strip() for part in line_clean.split('|')]
            
            if debug:
                print(f"  Parts: {parts}")
                print(f"  Length: {len(parts)}")
            
            # 跳过表头行和分隔符行，确保第一个字段是数字
            if len(parts) >= 8 and parts[0] != 'Epoch':
                try:
                    epoch = int(parts[0])
                    l_class = float(parts[1])
                    l_cons = float(parts[2])
                    l_domain = float(parts[3])
                    train_acc = float(parts[4])
                    test_acc = float(parts[5])
                    test_acc_ema = float(parts[6])
                    time_val = float(parts[7])
                    
                    data['epochs'].append(epoch)
                    data['train_loss_class'].append(l_class)
                    data['train_loss_consistency'].append(l_cons)
                    data['train_loss_domain'].append(l_domain)
                    data['train_accuracy'].append(train_acc)
                    data['test_accuracy'].append(test_acc)
                    data['test_accuracy_ema'].append(test_acc_ema)
                    data['time'].append(time_val)
                    
                    data_lines_found += 1
                    if debug and data_lines_found <= 3:
                        print(f"  Successfully parsed: epoch={epoch}, l_class={l_class}")
                    
                except (ValueError, IndexError) as e:
                    if debug:
                        print(f"  Parse error: {e}")
                    continue
    
    if debug:
        print(f"\nTotal data lines found: {data_lines_found}")
    
    # 查找最佳EMA准确率
    if data['test_accuracy_ema']:
        best_ema_acc = max(data['test_accuracy_ema'])
        config['best_ema_acc'] = best_ema_acc
    
    return data, config

def plot_training_curves(data, config, output_path=None):
    """
    绘制训练曲线 - 完全匹配train_hybrid.py的样式
    
    Args:
        data: 解析的训练数据
        config: 配置信息
        output_path: 输出路径，如果为None则自动生成
    """
    if not data['epochs']:
        print("Warning: No valid training data found")
        return
    
    # 创建图形 - 完全匹配train_hybrid.py
    fig = plt.figure(figsize=(20, 8))
    dataset = config.get('dataset', 'unknown')
    fig.suptitle(f'Training History for {dataset}')
    
    # 损失曲线 - 完全匹配train_hybrid.py
    ax1 = fig.add_subplot(121)
    ax1.set_title("Loss Curves")
    ax1.plot(data['train_loss_class'], label='L_Class (Train)')
    ax1.plot(data['train_loss_consistency'], label='L_Cons (Train)')
    ax1.plot(data['train_loss_domain'], label='L_Domain (Train)')
    ax1.legend()
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(bottom=0)
    
    # 准确率曲线 - 完全匹配train_hybrid.py
    ax2 = fig.add_subplot(122)
    ax2.set_title("Accuracy Curves")
    ax2.plot(data['train_accuracy'], label='Train Accuracy')
    ax2.plot(data['test_accuracy'], label='Test Acc (Student)')
    ax2.plot(data['test_accuracy_ema'], label='Test Acc (EMA/Teacher)')
    
    # 添加最佳准确率线 - 完全匹配train_hybrid.py
    if 'best_ema_acc' in config:
        best_acc = config['best_ema_acc']
        ax2.axhline(y=best_acc, color='g', linestyle='--', label=f'Best EMA Acc: {best_acc:.4f}')
    
    ax2.legend()
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(0, 1.0)
    
    # 保存图像
    if output_path is None:
        import datetime
        date = datetime.datetime.now().strftime("%Y%m%d")
        output_path = f"result/Final_Hybrid_{dataset}_{date}.png"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig.savefig(output_path)
    print(f"Result plot saved to {output_path}")
    
    # 打印统计信息
    print(f"\n=== Training Statistics ===")
    print(f"Dataset: {dataset}")
    print(f"Total epochs: {len(data['epochs'])}")
    if data['train_accuracy']:
        print(f"Final train accuracy: {data['train_accuracy'][-1]:.4f}")
    if data['test_accuracy']:
        print(f"Final test accuracy (Student): {data['test_accuracy'][-1]:.4f}")
    if data['test_accuracy_ema']:
        print(f"Final test accuracy (EMA): {data['test_accuracy_ema'][-1]:.4f}")
        print(f"Best EMA accuracy: {max(data['test_accuracy_ema']):.4f}")
    
    # 保存训练历史数据 - 匹配train_hybrid.py
    history = {
        'train_loss_class': data['train_loss_class'],
        'train_loss_consistency': data['train_loss_consistency'],
        'train_loss_domain': data['train_loss_domain'],
        'train_accuracy': data['train_accuracy'],
        'test_accuracy': data['test_accuracy'],
        'test_accuracy_ema': data['test_accuracy_ema']
    }
    
    if output_path:
        history_path = output_path.replace('.png', '_history.npy')
        np.save(history_path, history)
        print(f"Training history saved to {history_path}")
    
    plt.close(fig)  # 释放内存

def main():
    parser = argparse.ArgumentParser(description='绘制训练日志数据图表')
    parser.add_argument('--log-file', default='log_yandex.txt', type=str, 
                       help='日志文件路径 (默认: log_yandex.txt)')
    parser.add_argument('--output', default=None, type=str, 
                       help='输出图像路径 (默认: 自动生成)')
    parser.add_argument('--show-config', action='store_true', 
                       help='显示解析的配置信息')
    parser.add_argument('--debug', action='store_true', 
                       help='显示调试信息')
    
    args = parser.parse_args()
    
    try:
        # 解析日志文件
        print(f"正在解析日志文件: {args.log_file}")
        data, config = parse_log_file(args.log_file, debug=args.debug)
        
        if args.show_config:
            print("\n=== 解析的配置信息 ===")
            for key, value in config.items():
                print(f"{key}: {value}")
        
        # 绘制训练曲线
        plot_training_curves(data, config, args.output)
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 