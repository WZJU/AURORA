'''
Update date: 2026-05-10
Author: Wenzhe
Description:
1. 加入可学习的BOS和EOS
'''
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import math
from wingVAE import apply_norm, denormalize_geometry, MEAN, STD, SCALE_MEAN, SCALE_STD



class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建形状为 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        # 计算频率分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加一个 batch 维度，并注册为 buffer（不参与梯度更新，但会随模型移动到对应设备）
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        # x: (Batch, Seq_len, Embed_dim)
        # 根据输入序列的实际长度截取对应的位置编码并相加
        return x + self.pe[:, :x.size(1), :]

class WingArTransformer(pl.LightningModule):
    def __init__(self, pretrained_vae, latent_dim=32, embed_dim=128, pose_dim=3, learning_rate=0.001, dropout_rate=0.1, train_stage="geom"):
        super().__init__()
        self.register_buffer('STD', torch.tensor(STD, dtype=torch.float32))
        self.register_buffer('MEAN', torch.tensor(MEAN, dtype=torch.float32))
        self.register_buffer('SCALE_MEAN', torch.tensor(SCALE_MEAN, dtype=torch.float32))
        self.register_buffer('SCALE_STD', torch.tensor(SCALE_STD, dtype=torch.float32))
        self.register_buffer('AR_MEAN', torch.tensor(12, dtype=torch.float32))

        self.save_hyperparameters(ignore=['pretrained_vae'])
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.train_stage = train_stage
        
        # 1. 冻结预训练 VAE
        self.vae = pretrained_vae
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # 2. 约束参数 Embedding
        self.cond_embeddings = nn.ModuleDict({
            # 工况参数
            'mach': nn.Linear(1, embed_dim),
            'alpha_sin': nn.Linear(1, embed_dim),
            'alpha_cos': nn.Linear(1, embed_dim), 

            # 性能指标（均为可选）
            'cl': nn.Linear(1, embed_dim),
            'cd': nn.Linear(1, embed_dim),
            'cm': nn.Linear(1, embed_dim),

            # 全局几何与动态残差约束（新加入）
            'ar_target': nn.Linear(1, embed_dim), # 全局展弦比目标
            'b_res': nn.Linear(1, embed_dim),     # 剩余展长比例
            'S_res': nn.Linear(1, embed_dim),     # 剩余面积比例
        })
        
        # 缺失值的特殊嵌入向量
        self.missing_embedding = nn.Parameter(torch.randn(embed_dim))
        # 工况-性能融合层 
        self.condition_fuser = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 可学习的BOS和EOS
        self.BOS = nn.Parameter(torch.randn(1, latent_dim))
        self.EOS = nn.Parameter(torch.randn(1, latent_dim))
        self.trainable_pose = nn.Parameter(torch.randn(1, pose_dim))

        # 3. 序列特征处理
        self.z_embedding = nn.Linear(latent_dim, embed_dim)
        self.pose_embedding = nn.Linear(pose_dim, embed_dim)
        # 支持最大 100 个剖面的位置编码
        self.pos_embedding = SinusoidalPositionalEncoding(d_model=embed_dim, max_len=100)
        
        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # 5. 多任务输出头
        self.fc_z = nn.Linear(embed_dim, latent_dim)
        self.fc_pose = nn.Linear(embed_dim, pose_dim) # [dihedral(sin), dihedral(cos), scale_factor]
        self.fc_eos = nn.Linear(embed_dim, 1)

    def _embed_with_mask(self, val, emb_layer):
        """
        处理可能包含 NaN 的输入，为缺失值提供特殊嵌入
        
        Args:
            val: [B, num_conds, 1] - 可能包含 NaN
            emb_layer: 对应的嵌入层
        
        Returns:
            [B, num_conds, embed_dim]
        """
        # 标记有效位置（非 NaN）
        valid_mask = ~torch.isnan(val)  # [B, num_conds, 1]，输入为nan时，valid_mask为False
        
        # 用 0 填充 NaN，避免传播问题
        val_safe = torch.nan_to_num(val, nan=0.0)
        
        # 将数据转换为与模型权重相同的 dtype（处理 double/float 类型不匹配问题）
        val_safe = val_safe.to(emb_layer.weight.dtype)
        
        # 正常嵌入
        embedded = emb_layer(val_safe)  # [B, num_conds, embed_dim]
        
        # 对 NaN 位置使用特殊嵌入
        missing_emb = self.missing_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        missing_emb = missing_emb.expand(embedded.shape[0], embedded.shape[1], -1)  # [B, num_conds, D]
        
        # 组合：有效位置用正常嵌入，缺失位置用特殊嵌入
        result = torch.where(valid_mask.expand(-1, -1, embedded.shape[-1]), 
                            embedded, 
                            missing_emb)
        
        return result, valid_mask

    def _prepare_memory(self, cond_dict):
        """
        扩展版：支持任意数量的工况-性能组合，性能指标可选
        
        输入格式：
        cond_dict = {
            'span': [B, 1],                    # 可选
            'area': [B, 1],                    # 可选
            'conditions': [B, num_conds, 6],   # [sin(alpha), cos(alpha), mach, cl, cd, cm]，NaN 表示缺失
        }
        
            # 示例1：完整工况（含 cm）
            conditions_full = torch.tensor([
                [1, 0, 0.9, 0.5, 0.01, 0.02],   # sin(alpha), cos(alpha), mach, cl, cd, cm
                [1, 0, 0.8, 0.8, 0.02, 0.05],
            ])

            # 示例2：缺失 cm
            conditions_no_cm = torch.tensor([
                [1, 0, 0.9, 0.5, 0.5, 0.01, float('nan')],  # cm 缺失
                [1, 0, 0.9, 0.5, 0.6, 0.015, float('nan')],
            ])

            # 示例3：只提供 cl（cd 和 cm 都缺失）
            conditions_cl_only = torch.tensor([
                [1, 0, 0.9, 0.5, 0.5, float('nan'), float('nan')],
            ])

            # 组合使用
            cond_dict = {
                'span': torch.tensor([10.0]),
                'conditions': conditions_no_cm  # 使用缺失 cm 的工况
            }

        每个性能指标均可为 NaN（可选）
        """
        memory_list = []
        device = self.device
        
        # 1. 处理全局无量纲展弦比目标 [B, 1] -> 嵌入后扩展为 [B, 1, D]
        if 'ar_target' in cond_dict and cond_dict['ar_target'] is not None:
            ar_val = cond_dict['ar_target']
            # 处理0D张量（标量）的情况
            if ar_val.dim() == 0:
                ar_val = ar_val.unsqueeze(0).unsqueeze(0)  # [1, 1]
            elif ar_val.dim() == 1:
                ar_val = ar_val.unsqueeze(1)  # [B] -> [B, 1]
            # 全局变量占一个时间步
            memory_list.append(self.cond_embeddings['ar_target'](ar_val).unsqueeze(1))
        
        # 2. 处理工况-性能对（核心扩展）
        if 'ac' in cond_dict and cond_dict['ac'] is not None:
            conditions = cond_dict['ac'].to(device)  # [B, num_conds, 5]
            B, num_conds, _ = conditions.shape
            
            # 提取各参数（支持 NaN）
            alpha_sin = conditions[:, :, 0:1]    # [B, num_conds, 1] - sin(alpha)
            alpha_cos = conditions[:, :, 1:2]    # [B, num_conds, 1] - cos(alpha)
            mach = conditions[:, :, 2:3]   # [B, num_conds, 1]
            cl = conditions[:, :, 3:4]      # [B, num_conds, 1] - 可选（NaN）
            cd = conditions[:, :, 4:5]      # [B, num_conds, 1] - 可选（NaN）
            cm = conditions[:, :, 5:6]      # [B, num_conds, 1] - 可选（NaN）
            
            # 检查整行是否都是 NaN 或 mach/alpha 缺失
            # row_valid_mask: [B, num_conds, 1]，True 表示该行有效
            row_valid_mask = ~(torch.isnan(mach) | torch.isnan(alpha_sin) | torch.isnan(conditions).all(dim=-1, keepdim=True))
            
            # 嵌入各参数（自动处理 NaN）
            mach_emb, _ = self._embed_with_mask(mach, self.cond_embeddings['mach'])
            alpha_sin_emb, _ = self._embed_with_mask(alpha_sin, self.cond_embeddings['alpha_sin'])
            alpha_cos_emb, _ = self._embed_with_mask(alpha_cos, self.cond_embeddings['alpha_cos'])
            alpha_emb = (alpha_sin_emb + alpha_cos_emb) * 0.5  # 融合 sin 和 cos 嵌入
            cl_emb, cl_mask = self._embed_with_mask(cl, self.cond_embeddings['cl']) #[B, num_conds, D]， cl_mask为True时，cl_emb为正常嵌入，为False时，cl_emb为特殊嵌入
            cd_emb, cd_mask = self._embed_with_mask(cd, self.cond_embeddings['cd'])
            cm_emb, cm_mask = self._embed_with_mask(cm, self.cond_embeddings['cm'])
            
            # 融合工况上下文，并对无效行进行 mask
            condition_context = (mach_emb + alpha_emb) * 0.5  # [B, num_conds, D]
            
            # 动态融合性能指标：只融合提供的指标
            perf_embs = []
            perf_masks = []
            
            if cl_mask.any():  # 如果有任何 cl 有效
                perf_embs.append(cl_emb) # [B, num_conds, D]
                perf_masks.append(cl_mask) # [B, num_conds, 1]
            if cd_mask.any():
                perf_embs.append(cd_emb)
                perf_masks.append(cd_mask)
            if cm_mask.any():
                perf_embs.append(cm_emb)
                perf_masks.append(cm_mask)
            
            if perf_embs:
                # 对多个性能指标进行融合（考虑 mask）
                stacked_perf = torch.stack(perf_embs, dim=2)  # [B, num_conds, num_metrics, D]
                stacked_mask = torch.stack(perf_masks, dim=2)  # [B, num_conds, num_metrics, 1]
                
                # 只对有效的指标取平均
                masked_perf = stacked_perf * stacked_mask  # 无效位置置为 0
                sum_perf = masked_perf.sum(dim=2)  # [B, num_conds, D]
                count_valid = stacked_mask.sum(dim=2).clamp(min=1)  # 避免除 0
                combined_perf = sum_perf / count_valid  # [B, num_conds, D]
                
                # 融合工况和性能
                fused = self.condition_fuser(
                    torch.cat([condition_context, combined_perf], dim=-1) # [B, num_conds, D+D]
                ) # [B, num_conds, D]
                memory_list.append(fused)
        
        # 3. 处理独立的性能约束（无工况时）
        standalone_perf_keys = ['cl', 'cd', 'cm']
        for key in standalone_perf_keys:
            if key in cond_dict and cond_dict[key] is not None:
                if 'ac' not in cond_dict:
                    val = cond_dict[key]
                    if val.dim() == 1:
                        val = val.unsqueeze(1)
                    memory_list.append(self.cond_embeddings[key](val).unsqueeze(1))
        
        if not memory_list:
            return torch.zeros(1, 1, self.hparams.embed_dim).to(device)
        
        combined_memory = torch.cat(memory_list, dim=1)  # [B, N_AC + N_GEO, D+D]
        
        # 训练时对序列顺序进行随机打乱
        if self.training:
            B, N, D = combined_memory.shape
            # 确保索引在正确的设备上生成
            indices = torch.randperm(N, device=combined_memory.device).unsqueeze(0).expand(B, -1)  # [B, N]
            combined_memory = combined_memory.gather(1, indices.unsqueeze(-1).expand(-1, -1, D))
        
        return combined_memory
    '''
    def _z_fusion(self, z_seq, pose_in, cond_dict):
        tgt = self.z_embedding(z_seq)
        pose_emb = self.pose_embedding(pose_in)
        tgt = tgt + pose_emb
        return tgt
    '''
    def _z_fusion(self, z_seq, pose_in, cond_dict):
        # z_seq: [B, N-1, latent_dim]
        # pose_in: [B, N-1, pose_dim]
        current_len = z_seq.size(1)
        tgt = self.z_embedding(z_seq)     # [B, N-1, D]
        pose_emb = self.pose_embedding(pose_in) # [B, N-1, D]
        tgt = tgt + pose_emb
        
        # 核心：将一一对应的时序残差以“特征相加”或“拼接后映射”的方式融合进来
        # 此时 b_res[:, :-1, :] 长度和 tgt 完全一样，都是 N-1
        # 严格截取与当前输入序列完全等长的前 L 步，保证时序完全精准对齐
        if 'b_res' in cond_dict and cond_dict['b_res'] is not None:
            b_res_val = cond_dict['b_res'][:, :current_len, :].to(self.device) 
            b_res_emb = self.cond_embeddings['b_res'](b_res_val)      
            tgt = tgt + b_res_emb 
            
        if 'S_res' in cond_dict and cond_dict['S_res'] is not None:
            S_res_val = cond_dict['S_res'][:, :current_len, :].to(self.device) 
            S_res_emb = self.cond_embeddings['S_res'](S_res_val)      
            tgt = tgt + S_res_emb
            
        return tgt
    
    def forward(self, z_seq, pose_in, cond_dict, tgt_mask=None, tgt_padding_mask=None):
        memory = self._prepare_memory(cond_dict) #[B, n, embed_dim]
        tgt = self._z_fusion(z_seq, pose_in, cond_dict)
        # 加上位置编码
        tgt = self.pos_embedding(tgt) #[B, N-1, embed_dim]
        
        features = self.transformer_decoder(
            tgt, memory, 
            tgt_mask=tgt_mask,  # 因果掩码,[N-1, N-1]
            tgt_key_padding_mask=tgt_padding_mask  # 填充掩码,[B, N-1]
        )
        
        pose_out = self.fc_pose(features) #[B, N-1, 2]
        
        return {
            'z_next': self.fc_z(features),
            'dihedral': pose_out[:, :, 0:2], #[Batch, Sequence_Len, 2]
            # 'scale_factor': torch.sigmoid(pose_out[:, :, 2:3]) * 2.0, # 限制缩放上限
            'scale_factor': pose_out[:, :, 2:3], # [Batch, Sequence_Len, 1]，不限制缩放上限
            'eos_logit': self.fc_eos(features)
        }
    
    def _cal_continuity_loss(self, z_pred, scale_pred, active_mask):
        # 已知z_seq第0位是BOS，第1位是翼型引导，所以要忽略掉前两位的连续性约束。geom_pred的第0位对应翼型引导段，不作为连续性约束的一部分。
        if z_pred.size(1) > 2: # 至少要有3个剖面才能计算连续性约束
            geom_pred = self.vae.decode(z_pred[:, 1:, :].reshape(-1, self.hparams.latent_dim)).reshape(z_pred.size(0), z_pred.size(1) - 1, -1) #[B, N-2, 47]， 忽略第0位的翼型引导段
            # 把lambda恢复为原始尺度
            scaled_part = torch.pow(10, geom_pred[:, :, 40:41] * self.STD[40:41] + self.MEAN[40:41])  # 第40列为lg(\lambda),这里将其还原为\lambda
            geom_pred = torch.cat([
                geom_pred[:, :, :40],   # 前 40 维保持不变
                scaled_part,            # 中间 40:41 维使用缩放后的新值
                geom_pred[:, :, 41:]    # 后面的维度保持不变
            ], dim=2)

            # 提取配对参数（保持你的索引逻辑）
            inner_airfoil = geom_pred[:, :-1, 20:40]  # [B, N-3, 20] 外侧翼型
            outer_airfoil = geom_pred[:, 1:, 0:20]     # [B, N-3, 20] 内侧翼型

            pair_mask = active_mask[:, 1:-1] & active_mask[:, 2:]   # [B, N-3]
            num_pairs = pair_mask.sum() + 1e-8  # 防止除零

            # 翼型连续性损失 (20维)
            loss_cont_airfoil = (
                nn.functional.mse_loss(inner_airfoil, outer_airfoil, reduction='none')
                .mean(dim=-1) * pair_mask
            ).sum() / num_pairs


            # 计算弦长连续性损失 (1维)
            # 新的弦长连续性依赖K_{i-1}\lambda_{i-1} = K_{i}, 其中，\lambda对应41列
            # 注意：geom_pred 是从 z_pred[:, 1:, :] 解码得到的，所以需要跳过 scale_pred 的第一个位置来对齐
            inner_lambda = scaled_part[:, :-1, :]     # [B, N-3, 1]，对应 z_pred[:, 1:-1, :]
            inner_chord = inner_lambda * (scale_pred[:, 1:-1, :] * self.SCALE_STD + self.SCALE_MEAN) # K_{i-1} * \lambda_{i-1}
            outer_chord = (scale_pred[:, 2:, :] * self.SCALE_STD + self.SCALE_MEAN) # K_{i}

            loss_cont_chord = (
                nn.functional.mse_loss(inner_chord, outer_chord, reduction='none')
                .mean(dim=-1) * pair_mask
            ).sum() / num_pairs

            loss_continuity = 0.8 * loss_cont_airfoil + 0.2 * loss_cont_chord
        else:
            loss_continuity = torch.tensor(0.0, device=self.device)
        return loss_continuity

    
    def random_dropout(self, cond_dict, dropout_rate=0.1):
        for key in cond_dict:
            if key in ['ar_target']:
                cond_dict[key] = nn.functional.dropout(cond_dict[key], dropout_rate, training=self.training)
            elif key == 'ac':
                cond_dict[key] = torch.where(
                    torch.rand_like(cond_dict[key]) < dropout_rate,
                    torch.tensor(float('nan'), device=cond_dict[key].device),
                    cond_dict[key]
                )
        return cond_dict
    

    def _training_single_step(self, z_seq, target_pose, target_eos, cond_dict, tgt_mask, padding_mask, dropout_rate=0.1):
        '''
        _training_single_step 的 Docstring
        
        :param self: 说明
        :param z_seq: 说明
        :param target_pose: 说明
        :param target_eos: 说明
        :param cond_dict: {'span': [B, 1], 'area': [B, 1], 'mac': [B, 1], 'ac': [B, Nac, 5]}
        :param tgt_mask: 说明
        :param padding_mask: 说明
        :return: 说明
        :rtype: tuple[Tensor, Tensor, Tensor, Any, Any]
        '''
        cond_dict = self.random_dropout(cond_dict, dropout_rate=dropout_rate)

        outputs = self.forward(
            z_seq[:, :-1, :], 
            target_pose[:, :-1, :],
            cond_dict, 
            tgt_mask=tgt_mask,
            tgt_padding_mask=padding_mask[:, :-1]
        )
        
        # 对齐输出
        target_len = z_seq.size(1) - 1
        z_pred = outputs['z_next'][:, -target_len:, :]
        dihedral_pred = outputs['dihedral'][:, -target_len:, :]
        scale_pred = outputs['scale_factor'][:, -target_len:, :]
        eos_pred = outputs['eos_logit'][:, -target_len:, :]

        active_mask = ~padding_mask[:, 1:] 
        num_active = active_mask.sum()

        # Loss 计算
        loss_z = (nn.functional.mse_loss(z_pred, z_seq[:, 1:, :], reduction='none').mean(dim=-1) * active_mask).sum() / num_active
        
        pred_pose = torch.cat([dihedral_pred, scale_pred], dim=-1)
        loss_pose = (nn.functional.mse_loss(pred_pose, target_pose[:, 1:, :], reduction='none').mean(dim=-1) * active_mask).sum() / num_active

        loss_eos = (nn.functional.binary_cross_entropy_with_logits(eos_pred, target_eos[:, 1:, :], reduction='none').squeeze(-1) * active_mask).sum() / num_active

        eos_pred_class = (eos_pred > 0).float() # Sigmoid(x)>0.5 等价于 x>0
        eos_acc = (eos_pred_class == target_eos[:, 1:, :]).float()
        # 仅计算有效位置的准确率
        eos_acc = (eos_acc.squeeze(-1) * active_mask).sum() / num_active

        # 补充连续性约束：相邻剖面之间衔接部位的参数应该尽可能一致（可选）
        loss_continuity = self._cal_continuity_loss(z_pred, scale_pred, active_mask)
        # return loss_z, loss_pose, loss_eos, loss_continuity
        return loss_z, loss_pose, loss_eos, loss_continuity, eos_acc
        

    def training_step(self, batch, batch_idx):
        '''
        'z_seq': [B, N, 32],
        'cond': {'cl': [B, 1], 'cd': [B, 1], 'span': [B, 1], ...}, #旧版格式,不建议使用
        
        'conditions': {'conditions': [B, n, 5]},  # [mach, alpha, cl, cd, cm], # 多工况约束
                     'span': [B, 1], 'area': [B, 1], 'mac': [B, 1]}

        'target_pose': [B, N, 3], 上反角和缩放因子
        'target_eos': [B, N, 1],
        'padding_mask': [B, N] (True for padding positions)
        '''
        z_seq = batch['z_seq'] #latent
        cond_dict = batch['conditions'] #dict,约束条件
        target_pose = batch['target_pose'] #位姿信息，上反角和缩放因子
        target_eos = batch['target_eos'] #序列结束标签
        padding_mask = batch['padding_mask'] #填充掩码

        B = z_seq.size(0)

        # 构造包含BOS和EOS的训练序列
        train_z_BOS = self.BOS.expand(B, 1, -1).to(self.device)
        train_z_EOS = self.EOS.expand(B, 1, -1).to(self.device)
        train_BOS_pose = self.trainable_pose.expand(B, 1, -1).to(self.device)
        train_EOS_pose = self.trainable_pose.expand(B, 1, -1).to(self.device)
        train_eos_BOS = torch.tensor([[0]]).expand(B, 1, -1).to(self.device)
        train_eos_EOS = torch.tensor([[1]]).expand(B, 1, -1).to(self.device)

        train_bres_BOS = torch.ones(B, 1, 1, device=self.device)
        train_Sres_BOS = torch.ones(B, 1, 1, device=self.device)
        train_bres_EOS = torch.zeros(B, 1, 1, device=self.device)
        train_Sres_EOS = torch.zeros(B, 1, 1, device=self.device)

        z_seq = torch.cat([train_z_BOS, z_seq, train_z_EOS], dim=1).to(self.device)
        target_pose = torch.cat([train_BOS_pose, target_pose, train_EOS_pose], dim=1).to(self.device)
        target_eos = torch.cat([train_eos_BOS, target_eos, train_eos_EOS], dim=1).to(self.device)
        # 把对应的序列合并到cond_dict
        cond_dict['b_res'] = torch.cat([train_bres_BOS, cond_dict['b_res'], train_bres_EOS], dim=1).to(self.device)
        cond_dict['S_res'] = torch.cat([train_Sres_BOS, cond_dict['S_res'], train_Sres_EOS], dim=1).to(self.device)

        #padding_mask也要对应修改
        bos_mask = torch.zeros(B, 1, dtype=torch.bool, device=self.device)
        eos_mask = torch.zeros(B, 1, dtype=torch.bool, device=self.device)
        padding_mask = torch.cat([bos_mask, padding_mask, eos_mask], dim=1)

        # 因果掩码：输入长度为 N-1, 给到forward的长度就是 N-1，所以掩码大小是 (N-1, N-1)
        sz = z_seq.size(1) - 1
        tgt_mask = torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1).bool()
        
        
        # 不采用推挽训练
        # 开启推挽训练
        # 先进行几何训练
        if self.train_stage == "geom" or self.train_stage == "push-pull":
            goe_cond_dict = {
                'ar_target' : cond_dict.get('ar_target', None) - self.AR_MEAN,
                'b_res' : cond_dict.get('b_res', None),
                'S_res' : cond_dict.get('S_res', None)
            }
            loss_z, loss_pose, loss_eos, loss_continuity, eos_acc = self._training_single_step(z_seq, target_pose, target_eos, goe_cond_dict, tgt_mask, padding_mask, dropout_rate=self.dropout_rate)
            total_geo_loss = loss_z + 0.3 * loss_pose + 0.3 * loss_eos + 0.1 * loss_continuity

            self.log_dict({
                        "train_geo/loss_z": loss_z, 
                        "train_geo/loss_continuity": loss_continuity,
                        "train_geo/loss_pose": loss_pose,
                        "train_geo/loss_eos": loss_eos,
                        "train_geo/eos_acc": eos_acc,
                        }, prog_bar=True)
        
        # 进行气动输入训练
        if self.train_stage == "phys" or self.train_stage == "push-pull":
            ac_cond_dict = {
                'ac' : cond_dict.get('ac', None)
            }
            loss_z, loss_pose, loss_eos, loss_continuity, eos_acc = self._training_single_step(z_seq, target_pose, target_eos, ac_cond_dict, tgt_mask, padding_mask, dropout_rate=self.dropout_rate)
            total_ac_loss = loss_z + 0.3 * loss_pose + 0.3 * loss_eos + 0.1 * loss_continuity
            self.log_dict({
                        "train_ac/loss_z": loss_z, 
                        "train_ac/loss_continuity": loss_continuity,
                        "train_ac/loss_pose": loss_pose,
                        "train_ac/loss_eos": loss_eos,
                        "train_ac/eos_acc": eos_acc,
                        }, prog_bar=True)
        if self.train_stage == "push-pull":
            # 总损失
            ac_cond_dict = {
                'ac' : cond_dict.get('ac', None),
                'ar_target' : cond_dict.get('ar_target', None) - self.AR_MEAN,
                'b_res' : cond_dict.get('b_res', None),
                'S_res' : cond_dict.get('S_res', None)
            }
            loss_z, loss_pose, loss_eos, loss_continuity, eos_acc = self._training_single_step(z_seq, target_pose, target_eos, ac_cond_dict, tgt_mask, padding_mask, dropout_rate=self.dropout_rate)
            total_pp_loss = loss_z + 0.3 * loss_pose + 0.3 * loss_eos + 0.1 * loss_continuity
            self.log_dict({
                        "train_pp/loss_z": loss_z, 
                        "train_pp/loss_continuity": loss_continuity,
                        "train_pp/loss_pose": loss_pose,
                        "train_pp/loss_eos": loss_eos,
                        "train_pp/eos_acc": eos_acc,
                        }, prog_bar=True)
        
        if self.train_stage == "geom":
            # 总损失
            total_loss = total_geo_loss
            self.log_dict({
                        "train/total_loss": total_loss
                        }, prog_bar=True)
        elif self.train_stage == "phys":
            # 总损失
            total_loss = total_ac_loss
            self.log_dict({
                        "train/total_loss": total_loss
                        }, prog_bar=True)
        elif self.train_stage == "push-pull":
            # 总损失
            total_loss = 0.3*total_geo_loss + 0.3*total_ac_loss + 0.4*total_pp_loss
            self.log_dict({
                        "train/total_loss": total_loss
                        }, prog_bar=True)
        
        '''
        loss_z, loss_pose, loss_eos, loss_continuity = self._training_single_step(z_seq, target_pose, target_eos, cond_dict, tgt_mask, padding_mask)
        total_loss = loss_z + 0.3 * loss_pose + 0.3 * loss_eos + 0.1 * loss_continuity
        self.log_dict({
                        "train/total_loss": total_loss,
                       "train/loss_z": loss_z, 
                       "train/loss_continuity": loss_continuity,
                        "train/loss_pose": loss_pose,
                        "train/loss_eos": loss_eos
                       }, prog_bar=True)
        '''

        return total_loss
    

    def validation_step(self, batch, batch_idx):
        '''
        验证步骤：逻辑与 training_step 基本一致，但用于监控泛化能力
        '''
        # 1. 数据解包 (与训练一致)
        z_seq = batch['z_seq']
        cond_dict = batch['conditions']
        target_pose = batch['target_pose']
        target_eos = batch['target_eos']
        padding_mask = batch['padding_mask']
        
        B = z_seq.size(0)

        # 构造包含BOS和EOS的训练序列
        train_z_BOS = self.BOS.expand(B, 1, -1).to(self.device)
        train_z_EOS = self.EOS.expand(B, 1, -1).to(self.device)
        train_BOS_pose = self.trainable_pose.expand(B, 1, -1).to(self.device)
        train_EOS_pose = self.trainable_pose.expand(B, 1, -1).to(self.device)
        train_eos_BOS = torch.tensor([[0]]).expand(B, 1, -1).to(self.device)
        train_eos_EOS = torch.tensor([[1]]).expand(B, 1, -1).to(self.device)
        train_bres_BOS = torch.tensor([[1]]).expand(B, 1, -1).to(self.device)
        train_bres_EOS = torch.tensor([[0]]).expand(B, 1, -1).to(self.device)
        train_Sres_BOS = torch.tensor([[1]]).expand(B, 1, -1).to(self.device)
        train_Sres_EOS = torch.tensor([[0]]).expand(B, 1, -1).to(self.device)

        z_seq = torch.cat([train_z_BOS, z_seq, train_z_EOS], dim=1).to(self.device)
        target_pose = torch.cat([train_BOS_pose, target_pose, train_EOS_pose], dim=1).to(self.device)
        target_eos = torch.cat([train_eos_BOS, target_eos, train_eos_EOS], dim=1).to(self.device)
        # 把对应的序列合并到cond_dict
        cond_dict['b_res'] = torch.cat([train_bres_BOS, cond_dict['b_res'], train_bres_EOS], dim=1).to(self.device)
        cond_dict['S_res'] = torch.cat([train_Sres_BOS, cond_dict['S_res'], train_Sres_EOS], dim=1).to(self.device)

        #padding_mask也要对应修改
        bos_mask = torch.zeros(B, 1, dtype=torch.bool, device=self.device)
        eos_mask = torch.zeros(B, 1, dtype=torch.bool, device=self.device)
        padding_mask = torch.cat([bos_mask, padding_mask, eos_mask], dim=1)

        # 2. 构建因果掩码 (与训练一致)
        sz = z_seq.size(1) - 1
        # 防止空序列导致 triu 报错
        if sz <= 0:
            return None 
            
        tgt_mask = torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1).bool()

        # print('valid')
        # print(z_seq.shape)
        # print(target_pose.shape)
        # print(tgt_mask.shape)
        # print(padding_mask.shape)
        
        # 3. 前向传播 (与训练一致)
        # 注意：这里不需要手动 torch.no_grad()，Lightning 会自动处理
        
        valid_cond_dict = {}
        #针对训练阶段修改cond_dict
        if self.train_stage == "geom":
            valid_cond_dict['ar_target'] = cond_dict['ar_target'] - self.AR_MEAN
            valid_cond_dict['b_res'] = cond_dict['b_res']
            valid_cond_dict['S_res'] = cond_dict['S_res']
        elif self.train_stage == "phys":
            valid_cond_dict['ac'] = cond_dict['ac']
            valid_cond_dict['b_res'] = cond_dict['b_res']
            valid_cond_dict['S_res'] = cond_dict['S_res']
        elif self.train_stage == "push-pull":
            valid_cond_dict = cond_dict
            valid_cond_dict['ar_target'] = cond_dict['ar_target'] - self.AR_MEAN

        loss_z, loss_pose, loss_eos, loss_continuity, eos_acc = self._training_single_step(
                z_seq, target_pose, target_eos, valid_cond_dict, tgt_mask, padding_mask, dropout_rate=0.0)

        
        # 总 Loss (权重系数应与训练保持一致)
        total_loss = loss_z + 0.2 * loss_pose + 0.3 * loss_eos + 0.1 * loss_continuity

        self.log_dict({
            "val/total_loss": total_loss,
            "val/loss_z": loss_z,
            "val/loss_pose": loss_pose,
            "val/loss_eos": loss_eos,
            "val/eos_accuracy": eos_acc,
            "val/loss_continuity": loss_continuity
        }, prog_bar=True, sync_dist=True) # sync_dist=True 用于多卡训练时同步指标

        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # 使用 ReduceLROnPlateau 调度器，根据验证损失动态调整学习率
        # 优点：续训时不受 epoch 计数影响，始终根据当前损失调整
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',              # 最小化验证损失
            factor=0.5,              # 学习率衰减因子
            patience=10,             # 连续 10 个 epoch 损失不下降则衰减
            threshold=1e-4,          # 损失下降阈值
            min_lr=self.learning_rate * 0.01,  # 最小学习率
            verbose=True             # 打印学习率变化
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val/total_loss'  # 监控的指标
            }
        }

    @torch.no_grad()
    def generate_with_constraints(self, cond_dict, assigned_airfoil=None, max_steps=40):
        """
        仅提供 cond_dict 即可生成机翼。
        内置 BOS (Beginning of Sequence) 机制。
        assigned_airfoil: 一维list或tensor, 长度为20，前10位为上表面，后10位为下表面，数值为翼型参数，提供后会被编码成初始的 z 向量，替换掉 BOS 的全零向量
        
        # 使用示例
        cond_dict = {
            'span': torch.tensor([[10.0]]),
            'conditions': torch.tensor([[
                [0.8, 5.0, 0.5, 0.01, float('nan')],  # cm 可选
            ]])
        }
        """
        self.eval()
        device = next(self.parameters()).device
        if cond_dict is None or len(cond_dict) == 0: #cond_Dict的组装逻辑后续再核实一下
            batch_size = 1
            cond_dict = {} # 确保后面 get 不会出错
        else:
            batch_size = next(iter(cond_dict.values())).size(0)

        if 'conditions' in cond_dict:
            condition = cond_dict['conditions']
            # try:
            #     divisor = 3
            #     condition['ac'][:, -1] = condition['ac'][:, -1] / divisor #  对cm指标进行调整
            # except:
            #     pass
            cond_dict['conditions'] = condition.to(device)
        
        # 1. 物理量追踪
        current_span = torch.zeros(batch_size, 1).to(device)
        current_area = torch.zeros(batch_size, 1).to(device)

        # 修改输入指标    
        try:    
            target_span = cond_dict['span']
            target_area = cond_dict['area']

            C_ref = target_area / target_span
            AR_target = (target_span ** 2) / target_area
            while AR_target.dim() < 2:
                AR_target = AR_target.unsqueeze(0)
        except:
            AR_target = cond_dict.get('ar_target', torch.tensor(float(12)).to(device))
            C_ref = cond_dict.get('c_ref', torch.tensor(float(3)).to(device))
            target_span = AR_target * C_ref
            target_area = C_ref**2 * AR_target
            print(f'使用默认值的 AR_target 和 C_ref。设计指标AR_target: {AR_target}, span: {target_span}, area: {target_area}')
        log_AR_target = AR_target/0.564 - 8.95 - self.AR_MEAN # 对AR指标进行调整
        # print(AR_target.shape)
        # print(AR_target)
        # 确保 AR_target 至少是二维的 [B, 1]
        
        # 初始化推导公式中分子所需的累计求和项
        sum_b_K = torch.zeros(batch_size, 1).to(device)
        sum_shape_area = torch.zeros(batch_size, 1).to(device)
        
        # 2. 初始输入：自动创建 BOS (全零占位)
        # curr_z_seq 形状: [B, 序列长度, latent_dim]
        # curr_z_seq = torch.zeros(batch_size, 1, self.hparams.latent_dim).to(device)
        curr_z_seq = self.BOS.reshape(batch_size, 1, -1).to(device)
        curr_pose_seq = self.trainable_pose.expand(batch_size, 1, -1).to(device) #[B, 1, 3]
        cond_dict['b_res'] = torch.ones(batch_size, 1, 1, device=device)
        cond_dict['S_res'] = torch.ones(batch_size, 1, 1, device=device)
        cond_dict['ar_target'] = log_AR_target

        # 如果有指定翼型assigned_airfoil，则构建参数表，并用 VAE 编码得到初始的翼型引导向量。
        # '''
        if assigned_airfoil is not None:
            assigned_airfoil = torch.tensor(assigned_airfoil, dtype=torch.float32, device=device)
            assigned_airfoil = assigned_airfoil.repeat(2) # 内外站位相同
            plane_params = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32, device=device) # lg(\lambda), lg(\bar b), \Lambda(sin, cos), \epsilon(sin,cos)
            init_wing_params = torch.cat([assigned_airfoil, plane_params], dim=0).unsqueeze(0) # [1, 47]
            with torch.no_grad():
                init_wing_params = apply_norm(init_wing_params).to(torch.float32)
                airfoil_guidance = self.vae.encode(init_wing_params) # [1, 32]
            airfoil_guidance = airfoil_guidance.unsqueeze(1) # [1, 1, 32]
            airfoil_guidance = airfoil_guidance.repeat(batch_size, 1, 1).to(device) # [B, 1, 32],扩展到整个 Batch
            curr_z_seq = torch.cat([curr_z_seq, airfoil_guidance], dim=1)
            local_pose = torch.tensor([[[0.0, 1.0, (1.0-SCALE_MEAN)/(SCALE_STD + 1e-5)]]]).repeat(batch_size, 1, 1)  # [B, 1, 3]
            curr_pose_seq = torch.cat([curr_pose_seq, local_pose], dim=1) # [B, 2, 3]
            # print(local_pose, airfoil_guidance)

        else:
            # 如果没用指定翼型，就用BOS得到的翼型引导段
            output = self.forward(curr_z_seq, curr_pose_seq, cond_dict) # 给出BOS对应的位姿信息
            z_next = output['z_next'][:, -1, :]
            scale_f = output['scale_factor'][:, -1, :]  # [B, 1]
            dihedral_angle_tensor = output['dihedral'][:, -1, :] #[B, 2]
            # local_pose = torch.cat([dihedral_angle_tensor,scale_f], dim=1) #[B, 3]
            local_pose = torch.tensor([[[0.0, 1.0, (1.0-SCALE_MEAN)/(SCALE_STD + 1e-5)]]]).repeat(batch_size, 1, 1)  # [B, 1, 3]
            # print(local_pose, z_next)
            curr_z_seq = torch.cat([curr_z_seq, z_next.unsqueeze(1)], dim=1)
            curr_pose_seq = torch.cat([curr_pose_seq, local_pose], dim=1) # [B, 2, 3]
        
        generated_zs = []
        generated_geoms = [] # 存储每一步生成的几何参数，形状 [B, N, 47]
        generated_eos = []
        dihedral_list = []
        scale_factor_list = []
        # active_mask 记录 Batch 中哪些样本还在生成
        active_mask = torch.ones(batch_size, 1, dtype=torch.bool).to(device)

        # 在进入循环前，初始化残差的时序历史列表（与初始的 curr_z_seq 长度完全对齐）
        # 初始时 curr_z_seq 包含 [BOS] 或者是 [BOS, 翼型引导]，对应的初始剩余比例都是 1.0
        init_len = curr_z_seq.size(1)
        b_res_history = [torch.ones(batch_size, 1, 1, device=device) for _ in range(init_len)]
        S_res_history = [torch.ones(batch_size, 1, 1, device=device) for _ in range(init_len)]

        for i in range(max_steps):
            # print(i)
            # 1. 将历史的所有残差在时序维度（dim=1）拼接起来，形成和训练完全一致的 [B, current_len, 1] 轨迹
            current_b_res_seq = torch.cat(b_res_history, dim=1)  # [B, current_len, 1]
            current_S_res_seq = torch.cat(S_res_history, dim=1)  # [B, current_len, 1]

            step_cond_dict = {
                'ar_target': log_AR_target,
                'b_res': current_b_res_seq,
                'S_res': current_S_res_seq,
                'ac': cond_dict.get('ac', None) # 气动工况照常输入
            }

            # 2. 自回归推理：输入当前已有的所有 z 和严格对应的残差历史纵列
            output = self.forward(curr_z_seq, curr_pose_seq, step_cond_dict)
            
            # 只取最后一个时间步的预测结果
            z_next = output['z_next'][:, -1, :] # [B, 32]

            # 计算和恢复原始参数
            scale_f = output['scale_factor'][:, -1, :] # [B, 1], 合并用的缩放因子预测
            dihedral_angle_tensor = output['dihedral'][:, -1, :] #[B, 2]
            eos_prob = torch.sigmoid(output['eos_logit'][:, -1, :]) # [B, 1]
            local_pose = torch.cat([dihedral_angle_tensor,scale_f], dim=1) #[B, 3]
            scale_f = output['scale_factor'][:, -1, :]*self.SCALE_STD + self.SCALE_MEAN # [B, 1]，反归一化成真正的缩放系数
            
            # 提取上反角信息
            dihedral_sin = dihedral_angle_tensor[:, 0]
            dihedral_cos = dihedral_angle_tensor[:, 1]
            dihedral_angle = torch.rad2deg(torch.atan2(dihedral_sin, dihedral_cos)) # 计算上反角

            # 3. 物理还原与面积/展长计算 (基于 VAE)
            raw_geom = self.vae.decode(z_next) # 预训练 VAE 解码
            # 40:dy, 41:chord_in, 42:chord_out
            scaled_geom = raw_geom * self.STD + self.MEAN # 反归一化

            # 提取无量纲翼段展长 \bar{b}_i 和 梢根比 \lambda_i
            # (假设你已经按照方案一重新制作了数据集，40位代表梢根比，41位代表无量纲展长)
            lambda_i = torch.pow(10, scaled_geom[:, 40:41])
            bar_b_i = torch.pow(10, scaled_geom[:, 41:42]) 
            scaled_geom[:, 40:41] = lambda_i
            scaled_geom[:, 41:42] = bar_b_i

            # print(lambda_i, bar_b_i)
            # --- 【更新手稿公式中的累计求和项，用于下一步循环】 ---
            # 对应公式中的 \sum \bar{b}_i * K_i
            sum_b_K += bar_b_i * scale_f * 2
            # 对应公式中的 0.5 * \sum (1 + \lambda_i) * K_i^2 * \bar{b}_i
            sum_shape_area += 0.5 * (1.0 + lambda_i) * (scale_f ** 2) * bar_b_i * 2 # 乘2是因为算出来的b都是半展长

            actual_dy = bar_b_i * scale_f * C_ref 
            seg_area = 0.5 * (1.0 + lambda_i) * (scale_f ** 2) * bar_b_i * C_ref * C_ref # 对应 S_i 公式

            # 4. 【核心改动】：预测完 s_i 后，计算出最新的剩余残差 res_i
            # 这个最新的 res_i 将在“下一步循环”中，作为新预测出的 z_next (s_{i+1}) 对应的控制条件
            next_b_res = ((AR_target - sum_b_K) / AR_target).unsqueeze(-1)  # 确保形状为 [B, 1, 1]
            next_S_res = ((AR_target - sum_shape_area) / AR_target).unsqueeze(-1)
            # print(next_b_res, next_S_res)
            
            # 将最新的残差动态追加到历史队列中，保持与下一次循环的 curr_z_seq 长度完美同步
            b_res_history.append(next_b_res)
            S_res_history.append(next_S_res)

            # 5. 判定停止条件 (满足任一即停)
            # - 模型自发 EOS
            # - 达到展长目标
            # - 达到面积目标
            stop_signal = (eos_prob > 0.5) #| (current_span >= target_span) | (current_area >= target_area)
            active_mask = active_mask & ~stop_signal # 更新存活掩码
            # 如果全部样本都停了，直接退出
            if not active_mask.any():
                print('all samples are done')
                break
            
            # 仅对尚未停止的样本累加物理量、写入设计参数
            current_span += actual_dy * active_mask.float()
            current_area += seg_area * active_mask.float()
            
            generated_zs.append(z_next)
            generated_geoms.append(scaled_geom)
            scale_factor_list.append(scale_f)
            generated_eos.append(eos_prob)
            dihedral_list.append(dihedral_angle.cpu())
                
            # 将新预测的 z 拼接到输入序列中，准备下一步生成
            curr_z_seq = torch.cat([curr_z_seq, z_next.unsqueeze(1)], dim=1)
            curr_pose_seq = torch.cat([curr_pose_seq, local_pose.unsqueeze(1)], dim=1) # [B, 2, 3]
            
        # return torch.stack(generated_zs, dim=1), current_span, current_area
        return torch.stack(generated_geoms, dim=1), torch.stack(dihedral_list, dim=1), torch.stack(scale_factor_list, dim=1), C_ref, torch.stack(generated_eos, dim=1), current_span, current_area