import torch
import os

class WingDataCollector:
    def __init__(self):
        # 内部维护一个列表，存放每个样本字典
        self.samples = []

    def add_sample(self, z_seq, cond_dict, target_pose, target_eos):
        """
        每次调用添加一个样本。
        输入可以是 [N, 32] 这种形状的张量（没有 B 维度）
        """
        # 确保数据在 CPU 上，防止长期占用显存
        sample = {
            'z_seq': z_seq.detach().cpu(),           # [N, 32]
            'conditions': {k: v.detach().cpu() for k, v in cond_dict.items()}, # 单独指标[1],气动指标[n,5]
            'target_pose': target_pose.detach().cpu(), # [N, 3]
            'target_eos': target_eos.detach().cpu()      # [N, 1]
        }
        self.samples.append(sample)

    def save(self, file_path):
        """
        将目前收集到的所有样本保存为一个 .pt 文件
        最终形成的是一个长度为 B 的 List[Dict]
        """
        if len(self.samples) == 0:
            print("Warning: No samples to save.")
            return
            
        torch.save(self.samples, file_path)
        print(f"Successfully saved {len(self.samples)} samples to {file_path}")
        
    def clear(self):
        """清空缓存，以便开始收集下一组数据"""
        self.samples = []

# --- 使用示例 ---
# collector = WingDataCollector()
# for i in range(100):
#     # 模拟生成一个样本的数据
#     z = torch.randn(15, 32) 
#     cond = {'cl': torch.tensor([0.5])}
#     pose = torch.randn(15, 3)
#     eos = torch.zeros(15, 1)
#     
#     collector.add_sample(z, cond, pose, eos)
#
# collector.save("my_wing_dataset_part1.pt")


import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class WingDataset(Dataset):
    def __init__(self, data_path):
        """
        data_path: 存储 list 数据的 .pt 文件
        每个元素为: {'z_seq': [N, 32], 'conditions': {...}, 'target_pose': [N, 3], 'target_eos': [N, 1]}
        """
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

'''
def wing_collate_fn(batch):
    """
    动态补齐函数：处理一个 Batch 内 N 不一致的情况
    """
    # 1. 提取序列数据
    z_seqs = [item['z_seq'] for item in batch]           # 每个元素形状 [Ni, 32]
    target_pose = [item['target_pose'] for item in batch] # 每个元素形状 [Ni, 3]
    target_eoss = [item['target_eos'] for item in batch]    # 每个元素形状 [Ni, 1]
    
    # 2. 使用 pad_sequence 进行动态补齐
    # batch_first=True 保证输出形状为 [batch_size, max_N_in_batch, feature_dim]
    # padding_value=0.0 默认填充 0
    z_padded = pad_sequence(z_seqs, batch_first=True, padding_value=0.0)
    pose_padded = pad_sequence(target_pose, batch_first=True, padding_value=0.0)
    
    # 对于 EOS 标签，通常填充 1 (代表忽略) 或者 0，取决于你的 Loss 计算逻辑
    # 如果你使用了 CrossEntropy 且有 ignore_index，这里可以填 ignore_index 的值
    eoss_padded = pad_sequence(target_eoss, batch_first=True, padding_value=1.0)
    
    # 3. 构造 Cond 字典的 Batch
    # 假设 cond 里的每个值都是标量 [1]
    cond_keys = batch[0]['conditions'].keys()
    cond_batch = {}
    for key in cond_keys:
        val = torch.stack([item['conditions'][key] for item in batch])
        if val.dim() == 1: 
            val = val.unsqueeze(1) # 确保是 [B, 1] 而不是 [B]
        cond_batch[key] = val
    
    # 4. (可选) 构造 Padding Mask
    # Transformer 需要知道哪些位置是填充的，以避免计算注意力
    # mask 形状: [batch_size, max_N_in_batch]
    lengths = torch.tensor([s.size(0) for s in z_seqs])
    max_len = z_padded.size(1)
    # 生成掩码：有效位置为 False，填充位置为 True (适配 PyTorch Transformer 的 src_key_padding_mask)
    padding_mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    return {
        'z_seq': z_padded,
        'conditions': cond_batch,
        'target_pose': pose_padded,
        'target_eos': eoss_padded,
        'padding_mask': padding_mask  # 传入 forward 使用
    }
'''

def wing_collate_fn(batch):
    """扩展支持含 NaN 的可变长度 conditions"""

    # 设定AC 的固定列数
    FIXED_MAX_COLS = 6

    z_seqs = [item['z_seq'] for item in batch]
    target_pose = [item['target_pose'] for item in batch]
    target_eoss = [item['target_eos'] for item in batch]
    
    z_padded = pad_sequence(z_seqs, batch_first=True, padding_value=0.0)
    pose_padded = pad_sequence(target_pose, batch_first=True, padding_value=0.0)
    eoss_padded = pad_sequence(target_eoss, batch_first=True, padding_value=1.0)
    
    cond_batch = {}
    cond_keys = batch[0]['conditions'].keys()
    
    for key in cond_keys:
        vals = [item['conditions'][key] for item in batch]
        # vals = []
        # for item in batch:
        #     if key in item['conditions']:
        #         vals.append(item['conditions'][key])
        #     else:
        #         # 如果该样本缺少这个键，用 NaN 张量填充
        #         vals.append(torch.tensor(float('nan')).unsqueeze(0))
        
        if key == 'ac' or key == 'conditions':
            # 处理工况矩阵：确保统一为 [num_conds, 5] 格式
            # 缺失的性能指标用 NaN 填充
            processed_vals = []
            for v in vals:
                if v.dim() == 1:
                    v = v.unsqueeze(0)
                # 确保至少有 FIXED_MAX_COLS 列（alpha, mach, cl, cd, cm）
                current_cols = v.size(1)

                if current_cols < FIXED_MAX_COLS:
                    # 1. 列数不足：在维度 1 进行填充 (Padding)
                    padding = torch.full(
                        (v.size(0), FIXED_MAX_COLS - current_cols), 
                        float('nan'), 
                        device=v.device
                    )
                    v = torch.cat([v, padding], dim=1)
                elif current_cols > FIXED_MAX_COLS:
                    # 2. 列数过多：在维度 1 进行截断 (Truncate)
                    v = v[:, :FIXED_MAX_COLS]
                
                processed_vals.append(v)
            
            # pad_sequence 处理可变工况数量
            cond_batch['ac'] = pad_sequence(
                processed_vals,
                batch_first=True,
                padding_value=float('nan')  # NaN 填充
            )
        
        elif key in ['b_res', 'S_res']:
            # 由于 b_res 和 S_res 的维度是 [N, 1]，N 是可变的，不能用 torch.stack
            # 必须用 pad_sequence 沿时间步进行对齐填充
            # 填充值设定为 0.0，完美适应 Padding 区的掩码计算
            cond_batch[key] = pad_sequence(
                vals,
                batch_first=True,
                padding_value=0.0
            )

        else:
            val = torch.stack(vals)
            if val.dim() == 1:
                val = val.unsqueeze(1)
            cond_batch[key] = val
    
    # padding mask
    lengths = torch.tensor([s.size(0) for s in z_seqs])
    max_len = z_padded.size(1)
    padding_mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    return {
        'z_seq': z_padded,
        'conditions': cond_batch,
        'target_pose': pose_padded,
        'target_eos': eoss_padded,
        'padding_mask': padding_mask
    }

# --- 使用方法 ---
# dataset = WingDataset("wing_data.pt")
# train_loader = DataLoader(
#     dataset, 
#     batch_size=32, 
#     shuffle=True, 
#     collate_fn=wing_collate_fn  # 关键：传入动态补齐函数
# )