import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import numpy as np
import os
import shutil
from pathlib import Path
from datasetProcess import WingDataset, wing_collate_fn
from WingAutoregressiveTransformer import WingArTransformer
from wingVAE import WingVAE

# 回调函数：在恢复训练后控制学习率
class OverrideLRCallback(pl.Callback):
    def __init__(self, learning_rate, override_lr=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.override_lr = override_lr
    
    def on_train_start(self, trainer, pl_module):
        optimizer = pl_module.optimizers()
        if self.override_lr:
            # 使用指定的学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            print(f"Learning rate set to: {self.learning_rate}")
        else:
            # 沿用 checkpoint 中的学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Using learning rate from the checkpoint: {current_lr}")


class ResetEarlyStoppingCallback(pl.Callback):
    """
    重置 EarlyStopping 和 ModelCheckpoint 回调的状态，使其在恢复训练时不使用之前的早停信息和最佳分数。
    """
    def __init__(self, new_filename=None):
        super().__init__()
        self.new_filename = new_filename

    def on_train_start(self, trainer, pl_module):
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.EarlyStopping):
                callback.wait_count = 0
                if callback.mode == 'min':
                    callback.best_score = torch.tensor(float('inf'), device=pl_module.device)
                else:
                    callback.best_score = torch.tensor(-float('inf'), device=pl_module.device)
                callback.best_model_path = ""
                print("EarlyStopping state has been reset.")

            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                # 重置 best_score
                if callback.mode == 'min':
                    callback.best_score = torch.tensor(float('inf'), device=pl_module.device)
                else:
                    callback.best_score = torch.tensor(-float('inf'), device=pl_module.device)
                callback.best_model_path = ""
                
                # 如果指定了新的文件名格式，强制更新
                if self.new_filename:
                    callback.filename = self.new_filename
                    # 直接设置 filename_pattern，不使用私有方法
                    callback.filename_pattern = callback.filename
                    print(f"ModelCheckpoint filename updated to: {self.new_filename}")
                
                print("ModelCheckpoint best_score has been reset.")

# 假设 WingVAE 类已经按照之前的逻辑定义好了
# 这里直接写执行脚本

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

def train():
    # 0. 保存代码快照
    save_code_to_conf(args.exp_dir)

    # 1. 准备数据
    batch_size = args.batch_size
    full_dataset = WingDataset(args.dataset_path)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size  # 剩余部分给测试集
    # 设置 generator 保证每次划分结果一致
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(62) 
    )
    # 构建训练集的 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,      # 每个 batch 包含 4 个样本
        shuffle=True,               # 训练时打乱顺序
        collate_fn=wing_collate_fn, # 【核心】传入你自定义的合并函数
        num_workers=0,              # 数据读取进程数 (调试时建议设为0)
        pin_memory=True             # 加速 GPU 传输
    )

    # 构建验证集的 DataLoader (通常不需要 shuffle)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=wing_collate_fn, 
        pin_memory=True
    )
  

    # 2. 初始化模型
    # 先初始化训练好的VAE模型
    VAEmodel = WingVAE(
        input_dim=46, 
        latent_dim=32, 
        hidden_dim=256, 
        condition_dim=3,
        switch_epoch=200, 
        kl_coeff=0.005, 
        phys_coeff=1.5
    )
    model_path = Path(__file__).resolve().parents[1] / "checkpoints"
    checkpoint = torch.load(model_path / "vae_last.ckpt", map_location=torch.device('cpu'))
    VAEmodel.load_state_dict(checkpoint['state_dict'])
    VAEmodel.eval()

    # 初始化transformer模型
    AR_model = WingArTransformer(
        pretrained_vae=VAEmodel,
        latent_dim=32,
        embed_dim=128,
        learning_rate=args.learning_rate,
        train_stage=args.train_stage,
        dropout_rate=args.dropout_rate
        )
    
    # 用于断点续训的 checkpoint 路径
    resume_ckpt_path = args.resume_wight if args.resume_wight else None
      
    # 3. 设置训练器
    # pytorch lightning callbacks 
    if args.train_stage == "geom":
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.exp_dir,
            monitor='train/total_loss',
            filename='wing-transformer-geom-{epoch:02d}',
            save_top_k=1,
            mode='min',
            save_last=True
        )
    elif args.train_stage == "phys":
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.exp_dir,
            monitor='train/total_loss',
            filename='wing-transformer-phys-{epoch:02d}',
            save_top_k=1,
            mode='min',
            save_last=True
        )
    elif args.train_stage == "push-pull":
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.exp_dir,
            monitor='train/total_loss',
            filename='wing-transformer-push-pull-{epoch:02d}',
            save_top_k=1,
            mode='min',
            save_last=True
        )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # 如果恢复训练，添加回调函数控制学习率和重置早停状态
    callbacks = [checkpoint_callback, EarlyStopping(monitor="val/total_loss", patience=20), lr_monitor]
    if resume_ckpt_path:
        callbacks.append(OverrideLRCallback(learning_rate=args.learning_rate, override_lr=args.override_lr))
        # 传入当前训练阶段对应的文件名格式，确保恢复训练时使用新的命名格式
        filename_map = {
            "geom": 'wing-transformer-geom-{epoch:02d}',
            "phys": 'wing-transformer-phys-{epoch:02d}',
            "push-pull": 'wing-transformer-push-pull-{epoch:02d}'
        }
        callbacks.append(ResetEarlyStoppingCallback(new_filename=filename_map.get(args.train_stage)))

    trainer = pl.Trainer(
        max_epochs=400,
        accelerator="auto", # 自动检测 GPU/MPS/CPU
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        precision=32, # 如果对精度要求极高可设为 64
        default_root_dir=os.path.join("tensorboard_logs", args.exp_dir)
    )

    # 4. 开始训练
    # 注意：为了简化，我们可以直接分两次调用 fit，或者在一个 fit 里通过 epoch 判断。
    # 方案 A：一次 fit，内部判断 epoch (正如我们在 WingVAE 内部写的逻辑)
    # 我们需要一个合并的数据加载器或者简单的逻辑切换
    
    # 这里我们采用一个技巧：先用几何数据练满 100 轮，再手动切换
    print(">>> Start Training")
    trainer.fit(
        AR_model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,  # 传入几何验证集
        ckpt_path=resume_ckpt_path
    )

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", default=str(Path(__file__).resolve().parents[1] / "runs/transformer"),
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--dropout_rate", "-dr", default=0.1, type=float,
        help="dropout rate for transformer layers",
    )
    arg_parser.add_argument(
        "--learning_rate", "-l", default=0.0001, type=float,
        help="learning rate for transformer layers",
    )
    arg_parser.add_argument(
        "--override_lr", "-olr", action="store_true", default=True,
        help="whether to use the specified learning rate when resuming training (default: False, use checkpoint lr)",
    )
    #当 --override_lr 为 False （默认）时：沿用 checkpoint 中保存的学习率
    #当 --override_lr 为 True 时：使用 -l 参数指定的学习率
    arg_parser.add_argument(
        "--dataset_path", "-ds", default=str(Path(__file__).resolve().parents[1] / "data/train_dataV7.pt"),
        help="path to the training dataset (PyTorch .pt file)",
    )

    arg_parser.add_argument("--batch_size", "-b", default=64, type=int)

    arg_parser.add_argument("--resume_wight", "-rw", default=None, type=str,
        help="path to the resume weight file",
    )

    arg_parser.add_argument("--train_stage", "-ts", default="push-pull", type=str,
        help="train stage, geom or phys or push-pull",
    )

    args = arg_parser.parse_args()

    train()
