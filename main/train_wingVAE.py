import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
import pandas as pd
import torch
from wingVAE import WingVAE
import numpy as np
import os
import shutil
from pathlib import Path
from wingVAE import MEAN, STD, SCALE_MEAN, SCALE_STD, apply_norm, denormalize_geometry

# EXP01使用的归一化格式
# MEAN = np.array([0.17850959360039761, 0.10169849830937451, 0.24231527115193455, 0.0356391286760208, 0.3466380343564001, 0.016360906024927357, 0.3507674386912372, 0.10821011009496866, 0.21843688891535404, 0.14910861466478054, -0.1647817656329714, -0.07267581415468727, -0.2530387941398532, 0.0064062757917112715, -0.3458745438145741, -0.013725859545878573, -0.2643399768454656, -0.11611844440946628, -0.0216858440717235, 0.10033540381587398, 0.15792100630539838, 0.09297606076854259, 0.21017357801812203, 0.03717593662020186, 0.30285923728444714, 0.014812853648723045, 0.3185990934699495, 0.09349407627374215, 0.2186358500389968, 0.16544928334057193, -0.14203710549518275, -0.05941193872771065, -0.22256775119796607, 0.011480294089310292, -0.3020007967308457, -0.011732123862440741, -0.21865654844471633, -0.10268056671486467, 0.00885586120958862, 0.12267222297843708, 1.5148305595944336, 1.0001945537778711, 0.6686804601036092, 32.908861733378956, -2.4621365448660346])
# STD = np.array([0.051158038429098864, 0.027289125587100862, 0.08077143661619829, 0.11006394004319994, 0.19217508293770574, 0.211370015936046, 0.21126871397963748, 0.13782007720125755, 0.11865152192263625, 0.11149556504618585, 0.047555026428934916, 0.035702556135619005, 0.07115438139585371, 0.08795371602951732, 0.15879455422790323, 0.15318038963624908, 0.1708733304724696, 0.09831928129439799, 0.11629471834307903, 0.10488845225737198, 0.04983747612709268, 0.028260317740201053, 0.0725562708279452, 0.10414892256110664, 0.18009820662710868, 0.20054567400560286, 0.20485658905241766, 0.13147199812527358, 0.11659485487350249, 0.09932572278117821, 0.03916919721730589, 0.028426019555513517, 0.06501749000954672, 0.075509835266196, 0.13816483998839799, 0.1276099393742085, 0.13596350300246765, 0.08394811183206782, 0.0866080195585334, 0.08315919199835797, 0.4061599570714558, 0.49271881227736153, 0.3672267782938594, 4.681725126226871, 0.8246374378942508])

# EXP02使用的归一化格式
# MEAN = np.array([0.17850772354392538, 0.10169831821408365, 0.24231685040263015, 0.035641570641792186, 0.3466347380594082, 0.016372702629253, 0.35075318149766804, 0.10821736340905268, 0.21842443490870858, 0.14909535363391696, -0.16478094518487651, -0.07267785308869179, -0.253039553486658, 0.0064006190997018125, -0.34587130589288173, -0.013737857621760071, -0.2643323277612203, -0.11612509306460736, -0.021688433343986544, 0.1003295927450111, 0.15792031749376864, 0.09297780699380687, 0.21017802520276782, 0.03717889944170044, 0.30285860058284275, 0.014826462792721156, 0.3185890983122563, 0.09350162797446288, 0.21863171095733622, 0.16543658162917654, -0.14203841699495373, -0.059413732380533685, -0.222570429883248, 0.011474887803425338, -0.3020016009910544, -0.011742580198333206, -0.21865058975021026, -0.10268906958773606, 0.008859266903356524, 0.12266651277070112, 1.492087972877743, 0.907578363173255, 0.6007747529217831, 0.541455272629253, 0.8367665624774066, -0.04295580212913831, 0.9989733119545836])
# STD = np.array([0.05115652967417652, 0.027288090097289483, 0.08076855064442759, 0.11006008659591253, 0.19216810582193988, 0.21136624047847655, 0.21126692868914856, 0.1378173045755145, 0.1186555337359673, 0.1115016160465066, 0.04755330096462512, 0.03570195265918803, 0.07115171415871577, 0.08795273255484704, 0.15878891526165823, 0.15318067880305355, 0.17086903707350187, 0.09831846257342869, 0.11629064938523791, 0.10488654764891596, 0.04983563309747473, 0.028259942929172537, 0.07255527945765232, 0.1041454921193907, 0.18009133620051185, 0.20054404621912522, 0.20485194639023646, 0.13146980843567993, 0.11659135830580096, 0.09933254661379871, 0.039167986936876314, 0.02842567291751152, 0.06501572606405023, 0.07550947966282343, 0.13815958856727634, 0.12761066349527606, 0.1359600127986735, 0.08395053412577665, 0.08660558428533949, 0.08315857680732942, 0.5104456075806474, 0.2875799014971873, 0.22919960470842798, 0.06726380761840312, 0.0460813176983845, 0.014379144112609482, 0.0006010675482975929])

# 新版使用写死在wingVAE.py中

# 假设 WingVAE 类已经按照之前的逻辑定义好了
# 这里直接写执行脚本
def generate_latin_hypercube_samples(n_samples, param_bounds):
    """
    使用拉丁超立方采样生成几何参数样本
    
    Args:
        n_samples: 采样数量
        param_bounds: 参数边界，形状为 (2, n_params)，第0行是下限，第1行是上限
        n_params: 参数维度
    
    Returns:
        scaled_sample: 映射到参数边界后的样本，形状为 (n_samples, n_params)
    """    
    # 创建拉丁超立方采样器
    n_params = param_bounds.shape[1]
    sampler = qmc.LatinHypercube(d=n_params)
    sample = sampler.random(n=n_samples)
    
    # 将 [0,1] 区间映射到实际参数边界
    # param_bounds[0] 是所有参数的下限，param_bounds[1] 是所有参数的上限
    scaled_sample = qmc.scale(sample, param_bounds[0], param_bounds[1])
    
    return scaled_sample # 形状为 (n_samples, n_params)

def random_select_n_params(arr, n_samples, n_params):
    """
    从numpy数组中随机选择n_samples行
    
    Args:
        arr: 输入的numpy数组，形状为 (total_samples, n_params)
        n_samples: 需要选择的行数
        n_params: 参数维度（列数）
    
    Returns:
        selected: 随机选择的n_samples行数据，形状为 (n_samples, n_params)
    """
    # 确保不超过数组的行数
    n_samples = min(n_samples, arr.shape[0])
    
    # 使用numpy的随机选择，确保不重复
    indices = np.random.choice(arr.shape[0], size=n_samples, replace=False)
    selected = arr[indices, :n_params]
    
    return selected # 形状为 (n_samples, n_params)

def generate_train_param(n_samples, cst_array):
    """
    将几何参数转换为训练参数
    包括选择参数、把角度参数处理成sin和cos、整体的面积归一化、整体参数的归一化
    Args:
        n_samples: 采样数量
        cst_array: CST参数数组，形状为 (total_samples, 40)
    
    Returns:
        train_param: 训练参数张量，形状为 (n_samples, n_params)
    """
    # 产生CST参数
    cst_params = random_select_n_params(cst_array, n_samples, 40)
    param_bounds = np.array([
        [-1,  -1.3, -65, -5],#TR, bratio,	sweep	washout
        [1,   1.3, 70, 5] #
    ])
    # 考虑到TR和bratio的范围，直接使用对数进行归一化，上面生成的是对数空间的参数
    plane_params = generate_latin_hypercube_samples(n_samples, param_bounds)
    # print(plane_params)

    # 把角度参数处理成sin和cos
    # 正确：先将角度转换为弧度
    sweep_param = np.concatenate((np.sin(np.deg2rad(plane_params[:, 2:3])), np.cos(np.deg2rad(plane_params[:, 2:3]))), axis=1)
    washout_param = np.concatenate((np.sin(np.deg2rad(plane_params[:, 3:4])), np.cos(np.deg2rad(plane_params[:, 3:4]))), axis=1)
    #合并CST和平面参数
    train_params = np.concatenate((cst_params, plane_params[:, :2], sweep_param, washout_param), axis=1)
    train_param = apply_norm(train_params)
    
    # 转换为 torch 张量
    return torch.tensor(train_param, dtype=torch.float32)

def create_dataloader_from_samples(samples, batch_size=64, shuffle=True):
    """
    从样本张量创建 DataLoader
    
    Args:
        samples: 归一化后的样本张量，形状为 (n_samples, n_params)
        batch_size: 批次大小
        shuffle: 是否打乱
    
    Returns:
        DataLoader
    """
    # 创建数据集：几何参数, 空工况, 空指标
    dataset = TensorDataset(samples, 
                           torch.zeros(len(samples), 3),  # 空工况
                           torch.zeros(len(samples), 3))  # 空指标
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=4, pin_memory=True)

class WingDataset(Dataset):
    def __init__(self, csv_path, is_physics=False):
        """
        Args:
            csv_path: CSV文件路径
            mean_file: 预计算的均值npy文件或list (可选)
            std_file: 预计算的标准差npy文件或list (可选)
            is_physics: 是否为二阶段气动数据集
        """
        df = pd.read_csv(csv_path)
        self.is_physics = is_physics
        
        # 加载归一化参数 (如果是None则不进行归一化，建议始终提供)
        self.mean = MEAN
        self.std = STD

        if not is_physics:
            # 阶段一：纯几何数据 (假设CSV中只有46列几何参数)
            raw_data = df.values
            self.data = torch.tensor(self._apply_norm(raw_data), dtype=torch.float32)
        else:
            # 阶段二：全量数据 [几何(46), 工况(3), 指标(3)]
            # 1. 提取原始数据
            raw_geom = df.iloc[:, :46].values
            raw_cond = df.iloc[:, 46:49].values
            raw_phys = df.iloc[:, 49:52].values
            
            # 2. 归一化 (注意：通常工况和指标也需要各自的MEAN/STD，这里演示几何的归一化)
            self.geom = torch.tensor(self._apply_norm(raw_geom), dtype=torch.float32)
            self.cond = torch.tensor(raw_cond, dtype=torch.float32) # 工况视情况是否归一化
            self.phys = torch.tensor(raw_phys, dtype=torch.float32)

    def _apply_norm(self, x):
        """内部调用归一化，对几何参数进行归一化"""
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize_geometry(self, scaled_data):
        """
        外部调用：将模型输出的潜空间还原后的几何参数转回物理量纲
        输入可以是 Tensor 或 Numpy
        """
        if torch.is_tensor(scaled_data):
            scaled_data = scaled_data.detach().cpu().numpy()
        return scaled_data * self.std + self.mean

    def __len__(self):
        return len(self.data) if not self.is_physics else len(self.geom)

    def __getitem__(self, idx):
        if not self.is_physics:
            # 返回: 几何, 空工况, 空指标
            return self.data[idx], torch.zeros(3), torch.zeros(3)
        # 返回: 几何, 工况, 指标
        return self.geom[idx], self.cond[idx], self.phys[idx]

def copy_files(src, dst):
    """跨平台文件/目录复制"""
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists():
        raise FileNotFoundError(f"源路径不存在: {src}")
    
    if src_path.is_file():    
        shutil.copy2(src_path, dst_path)
    elif src_path.is_dir():
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        raise ValueError(f"未知路径类型: {src}")
    
def save_code_to_conf(conf_dir):
    # 创建基础目录
    code_dir = Path(conf_dir) / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    # 复制.py文件
    for py_file in Path(__file__).resolve().parent.glob("*.py"):
        try:
            copy_files(py_file, code_dir / py_file.name)
            print(f"[SUCCESS] 已复制 {py_file} -> {code_dir}")
        except Exception as e:
            print(f"[ERROR] 复制失败 {py_file}: {e}")
    
    # 复制.py文件
    for sh_file in Path(__file__).resolve().parent.glob("*.sh"):
        try:
            copy_files(sh_file, code_dir / sh_file.name)
            print(f"[SUCCESS] 已复制 {sh_file} -> {code_dir}")
        except Exception as e:
            print(f"[ERROR] 复制失败 {sh_file}: {e}")

def load_partial_checkpoint(model, ckpt_path, device="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    model_dict = model.state_dict()
    # 过滤掉 shape 不匹配的 key
    filtered_dict = {}
    skipped_keys = []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
        else:
            skipped_keys.append(k)

    # 更新权重
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print("✅ 已加载参数数:", len(filtered_dict))
    print("⚠️ 跳过参数数:", len(skipped_keys))
    if skipped_keys:
        print("跳过的 keys:", skipped_keys)

    return model

class WingLatentDataModule(pl.LightningDataModule):
    """
    使用拉丁超立方采样动态生成数据的数据模块
    每个epoch生成新的采样数据
    """
    def __init__(self, cst_array, train_samples=5000, val_samples=500, batch_size=64):
        super().__init__()
        self.cst_array = cst_array
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.batch_size = batch_size
        
        # 预先生成验证集（验证集保持不变以确保评估一致性）
        self.val_data = generate_train_param(self.val_samples, self.cst_array)
        
    def train_dataloader(self):
        # 每个epoch动态生成新的训练数据
        train_samples = generate_train_param(self.train_samples, self.cst_array)
        return create_dataloader_from_samples(train_samples, self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        # 验证集保持不变
        return create_dataloader_from_samples(self.val_data, self.batch_size, shuffle=False)

def train():
    # 0. 保存代码快照
    save_code_to_conf(args.exp_dir)

    # 1. 准备数据 - 使用拉丁超立方采样动态生成数据
    cst_array = np.loadtxt(Path(__file__).resolve().parents[1] / "data/Geo_combined.csv", delimiter=',', skiprows=1)
    cst_array = cst_array[:, :40]
    # valid_train_data = generate_train_param(2, cst_array)
    # # print(valid_train_data)
    dm = WingLatentDataModule(cst_array, train_samples=5000, val_samples=500, batch_size=args.batch_size)
    # 2. 初始化模型
    model = WingVAE(
        input_dim=46, 
        latent_dim=32, 
        hidden_dim=256, 
        condition_dim=3,
        switch_epoch=200, 
        kl_coeff=0.005, 
        phys_coeff=1.5
    )

    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = load_partial_checkpoint(model, resume, device="cpu")
    else:
        resume = None
      
    # 3. 设置训练器
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.exp_dir,
        monitor='train/geo_recon_loss',
        filename='wing-vae-{epoch:02d}',
        save_top_k=2,
        mode='min',
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="train/geo_recon_loss", patience=20), lr_monitor],
        enable_progress_bar=True,
        precision=32,
        default_root_dir=os.path.join("tensorboard_logs", args.exp_dir)
    )

    # 4. 开始训练
    print(">>> Starting Stage 1: Geometry Training with Latin Hypercube Sampling")
    trainer.fit(
        model, 
        datamodule=dm
    )

    # print(">>> Starting Stage 2: Physics Embedding")
    # # 调整 switch_epoch 确保进入第二阶段逻辑
    # trainer.fit(
    #     model, 
    #     train_dataloaders=phys_train_loader, 
    #     val_dataloaders=phys_val_loader  # 传入物理验证集
    # )

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", default=str(Path(__file__).resolve().parents[1] / "runs/vae"),
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument(
        "--context", "-c", default=False, type=bool,
        help="whether to use context information",
    )

    arg_parser.add_argument("--batch_size", "-b", default=64, type=int)
    args = arg_parser.parse_args()

    train()
