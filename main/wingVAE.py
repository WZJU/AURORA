import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

MEAN = np.array([0.17850772354392538, 0.10169831821408365, 0.24231685040263015, 0.035641570641792186, 0.3466347380594082, 0.016372702629253, 0.35075318149766804, 0.10821736340905268, 0.21842443490870858, 0.14909535363391696, -0.16478094518487651, -0.07267785308869179, -0.253039553486658, 0.0064006190997018125, -0.34587130589288173, -0.013737857621760071, -0.2643323277612203, -0.11612509306460736, -0.021688433343986544, 0.1003295927450111, 0.15792031749376864, 0.09297780699380687, 0.21017802520276782, 0.03717889944170044, 0.30285860058284275, 0.014826462792721156, 0.3185890983122563, 0.09350162797446288, 0.21863171095733622, 0.16543658162917654, -0.14203841699495373, -0.059413732380533685, -0.222570429883248, 0.011474887803425338, -0.3020016009910544, -0.011742580198333206, -0.21865058975021026, -0.10268906958773606, 0.008859266903356524, 0.12266651277070112, 0.0, 0.0 , 0.0, 0.04, 0.0, 0.04]) 
STD = np.array([0.05115652967417652, 0.027288090097289483, 0.08076855064442759, 0.11006008659591253, 0.19216810582193988, 0.21136624047847655, 0.21126692868914856, 0.1378173045755145, 0.1186555337359673, 0.1115016160465066, 0.04755330096462512, 0.03570195265918803, 0.07115171415871577, 0.08795273255484704, 0.15878891526165823, 0.15318067880305355, 0.17086903707350187, 0.09831846257342869, 0.11629064938523791, 0.10488654764891596, 0.04983563309747473, 0.028259942929172537, 0.07255527945765232, 0.1041454921193907, 0.18009133620051185, 0.20054404621912522, 0.20485194639023646, 0.13146980843567993, 0.11659135830580096, 0.09933254661379871, 0.039167986936876314, 0.02842567291751152, 0.06501572606405023, 0.07550947966282343, 0.13815958856727634, 0.12761066349527606, 0.1359600127986735, 0.08395053412577665, 0.08660558428533949, 0.08315857680732942, 0.5, 0.6, 0.2, 0.1, 0.2, 0.1])#最后6个值对应lg(\lambda), lg(\bar b), \Lambda(sin, cos), \epsilon(sin,cos)
# EXP02使用的归一化格式
SCALE_MEAN = 1.0  # 放大倍率的mean
SCALE_STD = 0.5   # 放大倍率的std

def apply_norm(x):
    """内部调用归一化，对几何参数进行归一化"""
    return (x - MEAN) / (STD + 1e-8)

def denormalize_geometry(scaled_data):
    """
    外部调用：将模型输出的潜空间还原后的几何参数转回物理量纲
    输入可以是 Tensor 或 Numpy
    """
    if torch.is_tensor(scaled_data):
        scaled_data = scaled_data.detach().cpu().numpy()
    return scaled_data * STD + MEAN

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)

class WingVAE(pl.LightningModule):
    def __init__(self, 
                 input_dim=46, 
                 latent_dim=32, 
                 hidden_dim=256, 
                 condition_dim=3,   # 例如: Ma, Re, Alpha
                 switch_epoch=500,
                 kl_coeff=0.001, 
                 phys_coeff=1.0):
        super().__init__()
        self.save_hyperparameters()
        
        # --- 编码器 ---
        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # --- 解码器 A: 几何还原 (核心：确保完美还原) ---
        self.decoder_geometry = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # --- 解码器 B: 气动力预测 (物理嵌入) ---
        # 输入是 潜变量 z + 外部工况条件
        self.decoder_physics = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # 输出: CL, CD, Cm
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --- 函数 1：第一阶段 - 仅几何重构训练 ---
    def _run_geometry_step(self, wing_params):
        hidden = self.encoder_backbone(wing_params)
        mu = self.fc_mu(hidden)
        logvar = self.fc_var(hidden)
        z = self.reparameterize(mu, logvar)
        
        pred_geometry = self.decoder_geometry(z)
        recon_loss = F.mse_loss(pred_geometry, wing_params)
        
        # 即使是纯几何阶段，也建议保留极小权重的 KL 以维持潜空间流形
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        total_loss = recon_loss + 0.0001 * kl_loss
        
        self.log('train/geo_recon_loss', recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        return total_loss

    # --- 函数 2：第二阶段 - 物理嵌入训练 ---
    def _run_physics_step(self, batch):
        wing_params, conditions, true_physics = batch
        
        # 编码
        hidden = self.encoder_backbone(wing_params)
        mu = self.fc_mu(hidden)
        logvar = self.fc_var(hidden)
        z = self.reparameterize(mu, logvar)
        
        # 几何重构 (在二阶段通常作为正则项，防止模型为了拟合物理而把几何学崩了)
        pred_geometry = self.decoder_geometry(z)
        recon_loss = F.mse_loss(pred_geometry, wing_params)
        
        # 气动预测
        z_cond = torch.cat([z, conditions], dim=1)
        pred_physics = self.decoder_physics(z_cond)
        phys_loss = F.mse_loss(pred_physics, true_physics)
        
        # KL 散度
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        # 二阶段综合损失
        total_loss = (recon_loss * 5.0 + 
                      kl_loss * self.hparams.kl_coeff + 
                      phys_loss * self.hparams.phys_coeff)
        
        self.log('train/phys_loss', phys_loss, on_step=True, on_epoch=True)
        self.log('train/total_phys_step_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        # 这里的 switch_epoch 可以根据你的需求动态设定
        if self.current_epoch < self.hparams.switch_epoch:
            # 只取参数进行几何训练
            wing_params = batch[0]
            return self._run_geometry_step(wing_params)
        else:
            # 运行全任务训练
            return self._run_physics_step(batch)
    
    def validation_step(self, batch, batch_idx):
        # 同样根据当前 epoch 判断验证逻辑
        if self.current_epoch < self.hparams.switch_epoch:
            # --- 阶段一验证：仅几何 ---
            wing_params = batch[0]
            hidden = self.encoder_backbone(wing_params)
            mu = self.fc_mu(hidden)
            logvar = self.fc_var(hidden)
            z = self.reparameterize(mu, logvar)
            
            pred_geometry = self.decoder_geometry(z)
            val_recon_loss = F.mse_loss(pred_geometry, wing_params)
            
            # 记录验证 Loss (sync_dist=True 用于分布式训练加速，单卡可不加)
            self.log('val/geo_recon_loss', val_recon_loss, prog_bar=True)
            return val_recon_loss
        else:
            # --- 阶段二验证：物理 + 几何 ---
            wing_params, conditions, true_physics = batch
            
            # 编码与重构
            hidden = self.encoder_backbone(wing_params)
            mu = self.fc_mu(hidden)
            logvar = self.fc_var(hidden)
            z = self.reparameterize(mu, logvar)
            
            pred_geometry = self.decoder_geometry(z)
            val_recon_loss = F.mse_loss(pred_geometry, wing_params)
            
            # 物理预测
            z_cond = torch.cat([z, conditions], dim=1)
            pred_physics = self.decoder_physics(z_cond)
            val_phys_loss = F.mse_loss(pred_physics, true_physics)
            
            # 记录物理验证 Loss
            self.log('val/phys_loss', val_phys_loss, prog_bar=True)
            self.log('val/geo_recon_loss', val_recon_loss) # 监控二阶段的几何保持能力
            
            return val_phys_loss

    def predict_performance(self, wing_params, conditions):
        """推理接口：给定几何参数和工况，预测性能"""
        mu = self.encode(wing_params)
        z_cond = torch.cat([mu, conditions], dim=1)
        return self.decoder_physics(z_cond)

    def encode(self, x):
        hidden = self.encoder_backbone(x)
        return self.fc_mu(hidden)
    
    def decode(self, z):
        return self.decoder_geometry(z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def decode_geometry(self, z):
        return self.decoder_geometry(z)