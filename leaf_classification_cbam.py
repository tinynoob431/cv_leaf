# # 叶片分类 CV：ResNet50 + CBAM 注意力机制增强版
#
# 这一版在原完整修复版基础上继续加入 **CBAM 注意力机制**，用于提升叶片细粒度分类中对叶缘、叶脉、纹理等关键区域的关注能力。
#
# 本 notebook 的重点改进：
#
# 1. 使用 Paddle 原生训练循环，不依赖 `paddlehub.Trainer`，避免原 notebook 中的 protobuf 兼容性问题。
# 2. 在 ResNet50 backbone 的最后一层卷积特征图之后加入 CBAM 模块，再进行全局池化和分类。
# 3. warmup 阶段冻结 ResNet50 backbone，只训练 CBAM + 分类头；fine-tune 阶段再整体微调。
# 4. 保留完整实验流程：数据检查、类别分布、训练曲线、混淆矩阵、每类准确率、错误样例、单图预测。
# 5. 方便写报告中的消融实验：可以与普通 ResNet50 baseline 的结果进行对比。
#
# 建议实验对比：
#
# | 实验 | 模型 | 说明 |
# |---|---|---|
# | Exp1 | ResNet50 | 普通迁移学习 baseline |
# | Exp2 | ResNet50 + 数据增强 | 加入随机裁剪、翻转、颜色扰动 |
# | Exp3 | ResNet50 + CBAM | 加入通道注意力和空间注意力 |
# | Exp4 | ResNet50 + CBAM + CutMix | 可选的进一步增强 |
#
# 运行方式：从上到下依次运行所有单元格即可。AI Studio 环境下一般不需要额外安装 Paddle。



# ## 0. 环境检查与全局配置



import os
import sys
import csv
import tarfile
import zipfile
import shutil
import random
import argparse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# 如果这一行报错，说明当前环境没有安装 paddle。
# AI Studio 一般已经预装；本地环境可以先安装 paddlepaddle 或 paddlepaddle-gpu。
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.io import Dataset, DataLoader

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)

print('Python:', sys.version)
print('Paddle:', paddle.__version__)
print('CUDA available:', paddle.device.is_compiled_with_cuda())




# ========== 可调参数 ==========
CFG = {
    'seed': 2026,
    'img_size': 224,
    'resize_size': 256,
    'batch_size': 16,
    'num_workers': 0,          # notebook 环境建议先用 0，稳定后可改成 2 或 4
    'warmup_epochs': 3,        # 只训练分类头
    'finetune_epochs': 20,     # 解冻后整体微调
    'lr_head': 1e-3,
    'lr_finetune': 1e-4,
    'weight_decay': 1e-4,
    'use_class_weight': True,  # 类别不均衡时更稳
    'pretrained': True,
    'use_cbam': True,          # 是否启用 CBAM 注意力机制
    'use_data_augmentation': True,  # 关闭后即 baseline 的确定性预处理
    'mixup_alpha': 0.0,        # <=0 表示关闭
    'cutmix_alpha': 0.0,       # <=0 表示关闭
    'mixup_cutmix_prob': 0.0,  # 每个 batch 执行 MixUp/CutMix 的概率
    'cbam_reduction': 16,      # CBAM 通道压缩比例
    'dropout': 0.2,            # 分类头 dropout，缓解过拟合
    'ablation_preset': 'cbam',
    'output_dir': 'outputs_leaf_ablation',
    'ablation_result_file': 'ablation_results.csv',
    'auto_prepare_dataset': True,
    'dataset_download_url': 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    'dataset_train_per_class': 120,
    'dataset_eval_per_class': 30,
}

ABLATION_PRESETS = {
    'baseline': {
        'report_model': 'ResNet50 baseline',
        'report_note': '普通迁移学习',
        'use_cbam': False,
        'use_data_augmentation': False,
        'mixup_alpha': 0.0,
        'cutmix_alpha': 0.0,
        'mixup_cutmix_prob': 0.0,
    },
    'aug': {
        'report_model': 'ResNet50 + 数据增强',
        'report_note': '提升泛化',
        'use_cbam': False,
        'use_data_augmentation': True,
        'mixup_alpha': 0.0,
        'cutmix_alpha': 0.0,
        'mixup_cutmix_prob': 0.0,
    },
    'cbam': {
        'report_model': 'ResNet50 + CBAM',
        'report_note': '引入注意力机制',
        'use_cbam': True,
        'use_data_augmentation': True,
        'mixup_alpha': 0.0,
        'cutmix_alpha': 0.0,
        'mixup_cutmix_prob': 0.0,
    },
    'cbam_mix': {
        'report_model': 'ResNet50 + CBAM + MixUp/CutMix',
        'report_note': '最终模型',
        'use_cbam': True,
        'use_data_augmentation': True,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'mixup_cutmix_prob': 0.6,
    },
}


def parse_runtime_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ablation', choices=list(ABLATION_PRESETS.keys()), default=CFG['ablation_preset'])
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--finetune_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--train_per_class', type=int, default=None)
    parser.add_argument('--eval_per_class', type=int, default=None)
    parser.add_argument('--disable_auto_prepare_dataset', action='store_true')
    args, _ = parser.parse_known_args()
    return args


args = parse_runtime_args()
CFG['ablation_preset'] = args.ablation
CFG.update(ABLATION_PRESETS[CFG['ablation_preset']])
if args.output_dir:
    CFG['output_dir'] = args.output_dir
if args.warmup_epochs is not None:
    CFG['warmup_epochs'] = max(0, int(args.warmup_epochs))
if args.finetune_epochs is not None:
    CFG['finetune_epochs'] = max(0, int(args.finetune_epochs))
if args.batch_size is not None:
    CFG['batch_size'] = max(1, int(args.batch_size))
if args.num_workers is not None:
    CFG['num_workers'] = max(0, int(args.num_workers))
if args.train_per_class is not None:
    CFG['dataset_train_per_class'] = max(1, int(args.train_per_class))
if args.eval_per_class is not None:
    CFG['dataset_eval_per_class'] = max(1, int(args.eval_per_class))
if args.disable_auto_prepare_dataset:
    CFG['auto_prepare_dataset'] = False

WORK_DIR = Path.cwd()
DATASET_DIR = WORK_DIR / 'processed_data'
output_root = Path(CFG['output_dir'])
if not output_root.is_absolute():
    output_root = WORK_DIR / output_root
OUTPUT_DIR = output_root / CFG['ablation_preset']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def seed_everything(seed=2026):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

seed_everything(CFG['seed'])

device = 'gpu' if paddle.device.is_compiled_with_cuda() else 'cpu'
paddle.set_device(device)
print('Using device:', device)
print('Work dir:', WORK_DIR)
print('Dataset dir:', DATASET_DIR)
print('Output dir:', OUTPUT_DIR)
print('Ablation preset:', CFG['ablation_preset'])
print('Model for report:', CFG['report_model'])
print('Epochs (warmup/finetune):', CFG['warmup_epochs'], CFG['finetune_epochs'])
print('Batch size:', CFG['batch_size'])




# ## 1. 数据准备与完整性检查
#
# 原始 notebook 直接写死了 `/home/aistudio/processed_data/label_list.txt`，这里改成相对路径，并自动处理 AI Studio 常见的数据压缩包位置。



def has_required_dataset_files():
    required = ['train.txt', 'eval.txt', 'label_list.txt']
    missing = [name for name in required if not (DATASET_DIR / name).exists()]
    return len(missing) == 0, missing


def prepare_dataset_from_tf_flowers(train_per_class, eval_per_class):
    print('\n开始自动准备公开数据集（TF Flowers）...')
    datasets_dir = WORK_DIR / 'datasets'
    datasets_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = datasets_dir / 'flower_photos.tgz'
    raw_root = datasets_dir / 'flower_photos'

    if not tgz_path.exists():
        print('下载数据集:', CFG['dataset_download_url'])
        urllib.request.urlretrieve(CFG['dataset_download_url'], str(tgz_path))
        print('下载完成:', tgz_path)
    else:
        print('已找到下载包:', tgz_path)

    if not raw_root.exists():
        print('解压数据集...')
        with tarfile.open(tgz_path, 'r:gz') as tf:
            try:
                tf.extractall(path=datasets_dir, filter='data')
            except TypeError:
                tf.extractall(path=datasets_dir)
        print('解压完成:', raw_root)
    else:
        print('已找到原始目录:', raw_root)

    classes = sorted([p.name for p in raw_root.iterdir() if p.is_dir()])
    if len(classes) < 5:
        raise RuntimeError(f'类别数量不足 5，当前仅 {len(classes)} 类: {classes}')
    print('类别列表:', classes)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    images_dir = DATASET_DIR / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(CFG['seed'])
    label_lines = []
    train_lines = []
    eval_lines = []

    need_per_class = int(train_per_class) + int(eval_per_class)
    for label_id, cls in enumerate(classes):
        src_images = sorted((raw_root / cls).glob('*.jpg'))
        rng.shuffle(src_images)
        if len(src_images) < need_per_class:
            raise RuntimeError(
                f'类别 {cls} 图片数不足，当前 {len(src_images)}，但需要 {need_per_class}（train+eval）。'
            )
        selected = src_images[:need_per_class]
        cls_train = selected[:train_per_class]
        cls_eval = selected[train_per_class:train_per_class + eval_per_class]

        dst_cls_dir = images_dir / cls
        dst_cls_dir.mkdir(parents=True, exist_ok=True)
        for p in cls_train + cls_eval:
            shutil.copy2(p, dst_cls_dir / p.name)

        label_lines.append(f'{label_id}\t{cls}')
        for p in cls_train:
            rel = (Path('images') / cls / p.name).as_posix()
            train_lines.append(f'{rel} {label_id}')
        for p in cls_eval:
            rel = (Path('images') / cls / p.name).as_posix()
            eval_lines.append(f'{rel} {label_id}')

    (DATASET_DIR / 'label_list.txt').write_text('\n'.join(label_lines) + '\n', encoding='utf-8')
    (DATASET_DIR / 'train.txt').write_text('\n'.join(train_lines) + '\n', encoding='utf-8')
    (DATASET_DIR / 'eval.txt').write_text('\n'.join(eval_lines) + '\n', encoding='utf-8')
    print(f'自动准备完成: train={len(train_lines)}, eval={len(eval_lines)}')


def ensure_dataset():
    """确保 processed_data 可用；优先使用本地数据，必要时自动下载并构建。"""
    ok, missing = has_required_dataset_files()
    if ok:
        print('processed_data 已存在且完整。')
        return

    if DATASET_DIR.exists():
        print('发现 processed_data 但文件不完整，缺少:', missing)

    zip_candidates = [
        WORK_DIR / 'data.zip',
        Path('/home/aistudio/data/data73970/data.zip'),
        Path('/home/aistudio/data/data.zip'),
    ]
    zip_path = next((p for p in zip_candidates if p.exists()), None)
    if zip_path is not None:
        print(f'正在解压: {zip_path}')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(WORK_DIR)

    ok, missing = has_required_dataset_files()
    if ok:
        print('数据集检查通过。')
        return

    if CFG.get('auto_prepare_dataset', True):
        prepare_dataset_from_tf_flowers(
            train_per_class=int(CFG['dataset_train_per_class']),
            eval_per_class=int(CFG['dataset_eval_per_class']),
        )
        ok, missing = has_required_dataset_files()
        if ok:
            print('数据集检查通过。')
            return

    raise FileNotFoundError(f'processed_data 缺少必要文件: {missing}')

ensure_dataset()




def read_label_list(label_path):
    """兼容以下格式：
    0\tclass_name
    0 class_name
    class_name
    """
    label_path = Path(label_path)
    raw = [line.strip() for line in label_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    indexed = []
    plain = []
    for line in raw:
        if '\t' in line:
            a, b = line.split('\t', 1)
        elif ' ' in line and line.split(' ', 1)[0].isdigit():
            a, b = line.split(' ', 1)
        else:
            a, b = None, line
        if a is not None and str(a).isdigit():
            indexed.append((int(a), b.strip()))
        else:
            plain.append(b.strip())
    if indexed:
        indexed = sorted(indexed, key=lambda x: x[0])
        labels = [name for _, name in indexed]
    else:
        labels = plain
    return labels


def read_split_file(txt_path):
    records = []
    txt_path = Path(txt_path)
    for line_no, line in enumerate(txt_path.read_text(encoding='utf-8').splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            img_rel, label = line.rsplit(' ', 1)
            records.append((img_rel, int(label)))
        except Exception as e:
            raise ValueError(f'{txt_path} 第 {line_no} 行格式错误: {line}') from e
    return records

label_list = read_label_list(DATASET_DIR / 'label_list.txt')
train_records = read_split_file(DATASET_DIR / 'train.txt')
eval_records = read_split_file(DATASET_DIR / 'eval.txt')
num_classes = len(label_list)

print('类别数:', num_classes)
print('训练集样本数:', len(train_records))
print('验证集样本数:', len(eval_records))
print('前 10 个类别:', label_list[:10])

# 标签范围检查
all_labels = [y for _, y in train_records + eval_records]
assert min(all_labels) >= 0, '存在负标签。'
assert max(all_labels) < num_classes, f'存在超出 label_list 范围的标签：max={max(all_labels)}, num_classes={num_classes}'

# 图片存在性检查
missing_imgs = []
for rel_path, _ in train_records + eval_records:
    if not (DATASET_DIR / rel_path).exists():
        missing_imgs.append(rel_path)
if missing_imgs:
    raise FileNotFoundError(f'存在缺失图片，前 10 个: {missing_imgs[:10]}')
else:
    print('图片路径检查通过。')




def count_by_class(records, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in records:
        counts[y] += 1
    return counts

train_counts = count_by_class(train_records, num_classes)
eval_counts = count_by_class(eval_records, num_classes)

if pd is not None:
    stat_df = pd.DataFrame({
        'label_id': np.arange(num_classes),
        'class_name': label_list,
        'train_count': train_counts,
        'eval_count': eval_counts,
        'total_count': train_counts + eval_counts,
    })
    display(stat_df.head(20))
    stat_df.to_csv(OUTPUT_DIR / 'class_distribution.csv', index=False, encoding='utf-8-sig')
else:
    print('pandas 不可用，直接打印前 20 类统计：')
    for i in range(min(20, num_classes)):
        print(i, label_list[i], train_counts[i], eval_counts[i])

print('训练集最大/最小类别样本数:', train_counts.max(), train_counts[train_counts > 0].min())
imbalance_ratio = train_counts.max() / max(1, train_counts[train_counts > 0].min())
print('类别不均衡比例 max/min:', round(float(imbalance_ratio), 2))

plt.figure(figsize=(12, 4))
plt.bar(np.arange(num_classes), train_counts, label='train')
plt.bar(np.arange(num_classes), eval_counts, bottom=train_counts, label='eval')
plt.xlabel('class id')
plt.ylabel('sample count')
plt.title('Class distribution')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=200)
plt.show()




def show_random_samples(records, max_show=12):
    sample_num = min(max_show, len(records))
    samples = random.sample(records, sample_num)
    cols = 4
    rows = int(np.ceil(sample_num / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, (rel_path, y) in enumerate(samples, start=1):
        img = Image.open(DATASET_DIR / rel_path).convert('RGB')
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{y}: {label_list[y]}', fontsize=9)
    plt.tight_layout()
    plt.show()

show_random_samples(train_records, max_show=12)




# ## 2. Dataset 与数据增强
#
# 训练集加入随机增强，验证集保持确定性预处理。这样可以减少过拟合，同时保证验证指标稳定。



def build_transforms():
    train_ops = []
    if CFG.get('use_data_augmentation', True):
        if hasattr(T, 'RandomResizedCrop'):
            train_ops.append(T.RandomResizedCrop(CFG['img_size'], scale=(0.75, 1.0), ratio=(0.8, 1.25)))
        else:
            train_ops.extend([T.Resize((CFG['resize_size'], CFG['resize_size'])), T.RandomCrop(CFG['img_size'])])

        if hasattr(T, 'RandomHorizontalFlip'):
            train_ops.append(T.RandomHorizontalFlip(prob=0.5))
        if hasattr(T, 'RandomVerticalFlip'):
            train_ops.append(T.RandomVerticalFlip(prob=0.2))
        if hasattr(T, 'ColorJitter'):
            train_ops.append(T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03))
    else:
        train_ops.extend([
            T.Resize((CFG['resize_size'], CFG['resize_size'])),
            T.CenterCrop(CFG['img_size']),
        ])

    train_ops.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_ops = [
        T.Resize((CFG['resize_size'], CFG['resize_size'])),
        T.CenterCrop(CFG['img_size']),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(train_ops), T.Compose(eval_ops)

train_transform, eval_transform = build_transforms()
print(train_transform)
print(eval_transform)




class LeafDataset(Dataset):
    def __init__(self, dataset_dir, split='train', transform=None, return_path=False):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform
        self.return_path = return_path

        if split == 'train':
            txt_name = 'train.txt'
        elif split in ['eval', 'val', 'valid', 'test']:
            txt_name = 'eval.txt'
        else:
            raise ValueError("split 只能是 'train' 或 'eval'")

        self.records = read_split_file(self.dataset_dir / txt_name)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rel_path, label = self.records[idx]
        img_path = self.dataset_dir / rel_path
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = np.int64(label)
        if self.return_path:
            return img, label, str(img_path)
        return img, label

train_dataset = LeafDataset(DATASET_DIR, split='train', transform=train_transform)
eval_dataset = LeafDataset(DATASET_DIR, split='eval', transform=eval_transform)
eval_dataset_with_path = LeafDataset(DATASET_DIR, split='eval', transform=eval_transform, return_path=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG['batch_size'],
    shuffle=True,
    num_workers=CFG['num_workers'],
    drop_last=False,
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=CFG['batch_size'],
    shuffle=False,
    num_workers=CFG['num_workers'],
    drop_last=False,
)

print('train batches:', len(train_loader))
print('eval batches:', len(eval_loader))




# ## 3. 模型：ResNet50 + CBAM 注意力机制
#
# 普通 ResNet50 会把最后一层卷积特征图直接做全局平均池化，然后进入全连接分类层。这里在全局池化之前加入 CBAM：
#
# - **Channel Attention**：判断哪些通道特征更重要，可以理解为增强叶脉、纹理、边缘等有用特征；
# - **Spatial Attention**：判断图像空间位置上哪里更重要，可以理解为让模型更关注叶片主体而不是背景。
#
# 实现位置：
#
# ```text
# 输入图像 -> ResNet50 卷积特征 -> CBAM -> 全局平均池化 -> Dropout -> 分类层
# ```
#
# warmup 阶段只训练 `CBAM + classifier`，后续 fine-tune 再训练整个网络。



class Identity(nn.Layer):
    def forward(self, x):
        return x


class ChannelAttention(nn.Layer):
    """CBAM 的通道注意力模块。

    输入:  [N, C, H, W]
    输出:  [N, C, 1, 1]，表示每个通道的重要性权重。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.mlp = nn.Sequential(
            nn.Conv2D(channels, hidden, kernel_size=1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(hidden, channels, kernel_size=1, bias_attr=False),
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return F.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Layer):
    """CBAM 的空间注意力模块。

    输入:  [N, C, H, W]
    输出:  [N, 1, H, W]，表示每个空间位置的重要性权重。
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2D(2, 1, kernel_size=kernel_size, padding=padding, bias_attr=False)

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        attn = paddle.concat([avg_out, max_out], axis=1)
        return F.sigmoid(self.conv(attn))


class CBAMBlock(nn.Layer):
    """Convolutional Block Attention Module。"""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ResNet50CBAM(nn.Layer):
    """ResNet50 + CBAM 叶片分类模型。

    为兼容不同 Paddle 版本中的 ResNet50 实现，forward_features 中同时支持两类结构：
    1. conv1/bn1/relu/maxpool/layer1~layer4；
    2. conv1/pool2d_max/blocks。
    """
    def __init__(self, num_classes, pretrained=True, reduction=16, dropout=0.2):
        super().__init__()
        try:
            self.backbone = paddle.vision.models.resnet50(pretrained=pretrained)
            if pretrained:
                print('已加载 ResNet50 预训练权重。')
        except Exception as e:
            print('预训练权重加载失败，改用随机初始化。错误信息：', repr(e))
            self.backbone = paddle.vision.models.resnet50(pretrained=False)

        # 避免原 ResNet 的 fc 参数参与训练或保存成无用参数。
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = Identity()

        self.cbam = CBAMBlock(channels=2048, reduction=reduction, spatial_kernel=7)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2048, num_classes)

    def forward_features(self, x):
        # 新版 torchvision 风格结构：conv1, bn1, relu, maxpool, layer1~layer4
        if all(hasattr(self.backbone, name) for name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']):
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            return x

        # Paddle 旧版常见结构：conv1, pool2d_max, blocks
        if hasattr(self.backbone, 'conv1') and hasattr(self.backbone, 'pool2d_max') and hasattr(self.backbone, 'blocks'):
            x = self.backbone.conv1(x)
            x = self.backbone.pool2d_max(x)
            for block in self.backbone.blocks:
                x = block(x)
            return x

        raise RuntimeError(
            '当前 Paddle 版本的 ResNet50 结构与本 notebook 不兼容。'
            '请打印 paddle.vision.models.resnet50(pretrained=False) 的结构后，按实际层名修改 forward_features。'
        )

    def forward(self, x):
        x = self.forward_features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class ResNet50Baseline(nn.Layer):
    """普通 ResNet50 baseline。保留这个类，方便你之后做消融实验。"""
    def __init__(self, num_classes, pretrained=True, dropout=0.2):
        super().__init__()
        try:
            self.backbone = paddle.vision.models.resnet50(pretrained=pretrained)
            if pretrained:
                print('已加载 ResNet50 预训练权重。')
        except Exception as e:
            print('预训练权重加载失败，改用随机初始化。错误信息：', repr(e))
            self.backbone = paddle.vision.models.resnet50(pretrained=False)
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 如果 backbone.fc 已经替换成 Identity，则 backbone 输出就是 2048 维特征。
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def build_model(num_classes):
    if CFG.get('use_cbam', True):
        print('构建模型：ResNet50 + CBAM')
        return ResNet50CBAM(
            num_classes=num_classes,
            pretrained=CFG['pretrained'],
            reduction=CFG['cbam_reduction'],
            dropout=CFG['dropout'],
        )
    print('构建模型：ResNet50 baseline')
    return ResNet50Baseline(
        num_classes=num_classes,
        pretrained=CFG['pretrained'],
        dropout=CFG['dropout'],
    )


model = build_model(num_classes)
print(model.__class__.__name__)




def set_backbone_trainable(model, trainable: bool):
    """trainable=False 时冻结 backbone，只训练 CBAM 和分类头；trainable=True 时全模型训练。"""
    for p in model.parameters():
        p.stop_gradient = not trainable

    # warmup 阶段仍然训练注意力模块和分类头。
    if hasattr(model, 'cbam'):
        for p in model.cbam.parameters():
            p.stop_gradient = False
    if hasattr(model, 'classifier'):
        for p in model.classifier.parameters():
            p.stop_gradient = False
    if hasattr(model, 'fc'):
        for p in model.fc.parameters():
            p.stop_gradient = False


def count_trainable_params(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(np.prod(p.shape))
        total += n
        if not p.stop_gradient:
            trainable += n
    return total, trainable

set_backbone_trainable(model, trainable=False)
total_params, trainable_params = count_trainable_params(model)
print(f'参数总量: {total_params:,}')
print(f'当前可训练参数量: {trainable_params:,}')
print('warmup 阶段训练内容: CBAM + classifier；fine-tune 阶段训练整个网络。')




# ## 4. 损失函数、训练与验证函数
#
# 加入：
#
# - 类别不均衡权重；
# - 训练 / 验证 loss、acc 记录；
# - 保存验证集最优模型；
# - 训练曲线保存。



def build_criterion():
    if CFG['use_class_weight']:
        counts = train_counts.astype('float32')
        counts[counts == 0] = 1.0
        weights = counts.sum() / (len(counts) * counts)
        weights = weights / weights.mean()
        weights = np.clip(weights, 0.2, 5.0).astype('float32')
        print('使用类别权重，范围:', float(weights.min()), float(weights.max()))
        return nn.CrossEntropyLoss(weight=paddle.to_tensor(weights, dtype='float32'))
    return nn.CrossEntropyLoss()

criterion = build_criterion()




def rand_bbox(width, height, lam):
    cut_ratio = np.sqrt(max(0.0, 1.0 - lam))
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = int(np.clip(cx - cut_w // 2, 0, width))
    y1 = int(np.clip(cy - cut_h // 2, 0, height))
    x2 = int(np.clip(cx + cut_w // 2, 0, width))
    y2 = int(np.clip(cy + cut_h // 2, 0, height))
    return x1, y1, x2, y2


def apply_mixup_or_cutmix(x, y):
    mixup_alpha = float(CFG.get('mixup_alpha', 0.0))
    cutmix_alpha = float(CFG.get('cutmix_alpha', 0.0))
    mix_prob = float(CFG.get('mixup_cutmix_prob', 0.0))

    if mix_prob <= 0 or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        return x, y, y, 1.0, None
    if np.random.rand() > mix_prob or x.shape[0] < 2:
        return x, y, y, 1.0, None

    batch_size = int(x.shape[0])
    perm = paddle.randperm(batch_size)
    x2 = paddle.gather(x, perm, axis=0)
    y2 = paddle.gather(y, perm, axis=0)

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        _, _, h, w = x.shape
        x1, y1, x2b, y2b = rand_bbox(width=int(w), height=int(h), lam=lam)
        if x2b <= x1 or y2b <= y1:
            return x, y, y, 1.0, None
        x[:, :, y1:y2b, x1:x2b] = x2[:, :, y1:y2b, x1:x2b]
        lam = 1.0 - ((x2b - x1) * (y2b - y1) / float(w * h))
        return x, y, y2, float(lam), 'cutmix'

    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
    x = lam * x + (1.0 - lam) * x2
    return x, y, y2, lam, 'mixup'


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for batch_id, (x, y) in enumerate(loader):
        x, y_a, y_b, lam, mix_method = apply_mixup_or_cutmix(x, y)
        logits = model(x)
        if mix_method is None:
            loss = criterion(logits, y_a)
        else:
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        pred = paddle.argmax(logits, axis=1)
        bs = y_a.shape[0]
        total_loss += float(loss.item()) * bs
        if mix_method is None:
            total_correct += float((pred == y_a).astype('float32').sum().item())
        else:
            correct_a = float((pred == y_a).astype('float32').sum().item())
            correct_b = float((pred == y_b).astype('float32').sum().item())
            total_correct += lam * correct_a + (1.0 - lam) * correct_b
        total_samples += bs

    return total_loss / total_samples, total_correct / total_samples


@paddle.no_grad()
def evaluate(model, loader, criterion, return_detail=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []

    for x, y in loader:
        logits = model(x)
        loss = criterion(logits, y)
        pred = paddle.argmax(logits, axis=1)
        bs = y.shape[0]

        total_loss += float(loss.item()) * bs
        total_correct += int((pred == y).astype('int64').sum().item())
        total_samples += bs

        if return_detail:
            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.numpy().tolist())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    if return_detail:
        return avg_loss, avg_acc, np.array(y_true), np.array(y_pred)
    return avg_loss, avg_acc




GLOBAL_BEST_ACC = -1.0

def run_stage(model, stage_name, epochs, lr, train_backbone):
    global GLOBAL_BEST_ACC

    set_backbone_trainable(model, trainable=train_backbone)
    total_params, trainable_params = count_trainable_params(model)
    print(f'\n===== {stage_name} =====')
    print(f'可训练参数: {trainable_params:,} / {total_params:,}')

    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=max(1, epochs))
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=[p for p in model.parameters() if not p.stop_gradient],
        weight_decay=CFG['weight_decay'],
    )

    stage_history = []
    best_path = OUTPUT_DIR / 'best_model.pdparams'

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion)
        lr_scheduler.step()

        row = {
            'stage': stage_name,
            'epoch': epoch,
            'lr': float(optimizer.get_lr()),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
        }
        stage_history.append(row)

        improved = eval_acc > GLOBAL_BEST_ACC
        if improved:
            GLOBAL_BEST_ACC = eval_acc
            paddle.save({
                'model_state': model.state_dict(),
                'label_list': label_list,
                'cfg': CFG,
                'best_eval_acc': float(eval_acc),
                'stage': stage_name,
                'epoch': epoch,
            }, str(best_path))

        mark = ' *best*' if improved else ''
        print(
            f"[{stage_name}] epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"eval_loss={eval_loss:.4f}, eval_acc={eval_acc:.4f}{mark}"
        )
    return stage_history




# 正式训练：先只训练分类头，再解冻全模型微调。
history = []

history += run_stage(
    model,
    stage_name='head_only',
    epochs=CFG['warmup_epochs'],
    lr=CFG['lr_head'],
    train_backbone=False,
)

history += run_stage(
    model,
    stage_name='finetune_all',
    epochs=CFG['finetune_epochs'],
    lr=CFG['lr_finetune'],
    train_backbone=True,
)

history_df = pd.DataFrame(history) if pd is not None else None
if history_df is not None:
    display(history_df.tail())
    history_df.to_csv(OUTPUT_DIR / 'train_history.csv', index=False, encoding='utf-8-sig')

print('训练结束，最优模型保存在:', OUTPUT_DIR / 'best_model.pdparams')




# ## 5. 训练曲线与最终评估



if history_df is not None and len(history_df) > 0:
    plt.figure(figsize=(8, 4))
    plt.plot(history_df.index + 1, history_df['train_loss'], label='train_loss')
    plt.plot(history_df.index + 1, history_df['eval_loss'], label='eval_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'loss_curve.png', dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(history_df.index + 1, history_df['train_acc'], label='train_acc')
    plt.plot(history_df.index + 1, history_df['eval_acc'], label='eval_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_curve.png', dpi=200)
    plt.show()




# 加载验证集最优模型，再做最终评估
best_path = OUTPUT_DIR / 'best_model.pdparams'
if best_path.exists():
    ckpt = paddle.load(str(best_path))
    model.set_state_dict(ckpt['model_state'])
    print('已加载最优模型。best_eval_acc:', ckpt.get('best_eval_acc'))
else:
    print('没有找到 best_model.pdparams，将使用当前模型评估。')

final_loss, final_acc, y_true, y_pred = evaluate(model, eval_loader, criterion, return_detail=True)
print(f'Final eval loss: {final_loss:.4f}')
print(f'Final eval acc : {final_acc:.4f}')




def load_ablation_results(csv_path):
    if not csv_path.exists():
        return []
    if pd is not None:
        df = pd.read_csv(csv_path)
        return df.to_dict('records')
    with csv_path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_ablation_results(csv_path, rows):
    if pd is not None:
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('timestamp')
            df = df.drop_duplicates(subset=['ablation_preset'], keep='last')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return
    if not rows:
        return
    keys = list(rows[0].keys())
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def update_ablation_table(final_acc, final_loss):
    csv_path = WORK_DIR / CFG['ablation_result_file']
    rows = load_ablation_results(csv_path)
    rows = [r for r in rows if r.get('ablation_preset') != CFG['ablation_preset']]

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ablation_preset': CFG['ablation_preset'],
        'model': CFG['report_model'],
        'note': CFG['report_note'],
        'use_cbam': bool(CFG['use_cbam']),
        'use_data_augmentation': bool(CFG['use_data_augmentation']),
        'mixup_alpha': float(CFG['mixup_alpha']),
        'cutmix_alpha': float(CFG['cutmix_alpha']),
        'mixup_cutmix_prob': float(CFG['mixup_cutmix_prob']),
        'final_eval_loss': float(final_loss),
        'final_eval_acc': float(final_acc),
    }
    rows.append(row)
    save_ablation_results(csv_path, rows)

    print('\n已更新消融结果文件:', csv_path)
    if pd is not None:
        result_df = pd.DataFrame(load_ablation_results(csv_path))
        display(result_df.sort_values('ablation_preset'))

        preset_order = ['baseline', 'aug', 'cbam', 'cbam_mix']
        preset_to_acc = {r['ablation_preset']: float(r['final_eval_acc']) for r in result_df.to_dict('records')}
        report_table = pd.DataFrame([
            {'模型': ABLATION_PRESETS[k]['report_model'], '验证准确率': preset_to_acc.get(k, np.nan), '说明': ABLATION_PRESETS[k]['report_note']}
            for k in preset_order
        ])
        display(report_table)
        report_table.to_csv(OUTPUT_DIR / 'ablation_report_table.csv', index=False, encoding='utf-8-sig')
    else:
        print('当前环境无 pandas，已写入 CSV，可手工整理成报告表格。')


update_ablation_table(final_acc, final_loss)


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

cm = confusion_matrix_np(y_true, y_pred, num_classes)
per_class_acc = np.divide(
    np.diag(cm),
    cm.sum(axis=1),
    out=np.zeros(num_classes, dtype=np.float64),
    where=cm.sum(axis=1) != 0,
)

if pd is not None:
    per_class_df = pd.DataFrame({
        'label_id': np.arange(num_classes),
        'class_name': label_list,
        'eval_count': cm.sum(axis=1),
        'correct': np.diag(cm),
        'per_class_acc': per_class_acc,
    }).sort_values('per_class_acc')
    display(per_class_df.head(15))
    per_class_df.to_csv(OUTPUT_DIR / 'per_class_accuracy.csv', index=False, encoding='utf-8-sig')

plt.figure(figsize=(max(10, num_classes * 0.35), max(8, num_classes * 0.35)))
plt.imshow(cm)
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.xticks(np.arange(num_classes), np.arange(num_classes), rotation=90)
plt.yticks(np.arange(num_classes), np.arange(num_classes))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=200)
plt.show()




# ## 6. 错误样例分析
#
# 这一步是报告里最有用的部分之一：可以直接看模型错在哪里，是背景干扰、光照问题、叶片遮挡，还是类别本身太相似。



def collate_with_path(batch):
    imgs, labels, paths = zip(*batch)
    imgs = paddle.stack([img if isinstance(img, paddle.Tensor) else paddle.to_tensor(img) for img in imgs])
    labels = paddle.to_tensor(labels, dtype='int64')
    return imgs, labels, list(paths)

@paddle.no_grad()
def collect_wrong_samples(model, dataset, max_keep=64):
    loader = DataLoader(
        dataset,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_with_path,
    )
    model.eval()
    wrong = []
    for x, y, paths in loader:
        logits = model(x)
        probs = F.softmax(logits, axis=1)
        conf = paddle.max(probs, axis=1)
        pred = paddle.argmax(probs, axis=1)
        y_np = y.numpy()
        pred_np = pred.numpy()
        conf_np = conf.numpy()
        for true_id, pred_id, score, path in zip(y_np, pred_np, conf_np, paths):
            if int(true_id) != int(pred_id):
                wrong.append({
                    'path': path,
                    'true_id': int(true_id),
                    'pred_id': int(pred_id),
                    'confidence': float(score),
                    'true_name': label_list[int(true_id)],
                    'pred_name': label_list[int(pred_id)],
                })
    wrong = sorted(wrong, key=lambda x: x['confidence'], reverse=True)
    return wrong[:max_keep]

wrong_samples = collect_wrong_samples(model, eval_dataset_with_path, max_keep=64)
print('验证集错误样例数量（最多保留 64 个）:', len(wrong_samples))

if pd is not None and wrong_samples:
    wrong_df = pd.DataFrame(wrong_samples)
    display(wrong_df.head(20))
    wrong_df.to_csv(OUTPUT_DIR / 'wrong_samples.csv', index=False, encoding='utf-8-sig')




def show_wrong_samples(wrong_samples, max_show=16):
    if not wrong_samples:
        print('没有错误样例可展示。')
        return
    sample_num = min(max_show, len(wrong_samples))
    cols = 4
    rows = int(np.ceil(sample_num / cols))
    plt.figure(figsize=(cols * 3.5, rows * 3.5))
    for i, item in enumerate(wrong_samples[:sample_num], start=1):
        img = Image.open(item['path']).convert('RGB')
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis('off')
        title = (
            f"T:{item['true_id']} {item['true_name']}\n"
            f"P:{item['pred_id']} {item['pred_name']}\n"
            f"conf={item['confidence']:.2f}"
        )
        plt.title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'wrong_samples.png', dpi=200)
    plt.show()

show_wrong_samples(wrong_samples, max_show=16)




# ## 7. 单张图片预测函数
#
# 训练结束后，可以用这个函数预测任意一张叶片图片。



@paddle.no_grad()
def predict_one(image_path, topk=5, model_path=None):
    image_path = Path(image_path)
    if model_path is not None:
        ckpt = paddle.load(str(model_path))
        model.set_state_dict(ckpt['model_state'])

    model.eval()
    img = Image.open(image_path).convert('RGB')
    x = eval_transform(img)
    if not isinstance(x, paddle.Tensor):
        x = paddle.to_tensor(x)
    x = x.unsqueeze(0)

    logits = model(x)
    probs = F.softmax(logits, axis=1)[0]
    values, indices = paddle.topk(probs, k=min(topk, num_classes))

    results = []
    for score, idx in zip(values.numpy().tolist(), indices.numpy().tolist()):
        results.append({
            'label_id': int(idx),
            'class_name': label_list[int(idx)],
            'probability': float(score),
        })
    return results

# 示例：把路径换成你要预测的图片
# predict_one(DATASET_DIR / train_records[0][0], topk=5, model_path=OUTPUT_DIR / 'best_model.pdparams')




# ## 8. 本 notebook 输出文件
#
# 运行完成后，`outputs_leaf_ablation/<ablation_preset>/` 下会保存：
#
# - `best_model.pdparams`：验证集最优模型；
# - `train_history.csv`：训练日志；
# - `loss_curve.png`：loss 曲线；
# - `accuracy_curve.png`：accuracy 曲线；
# - `class_distribution.csv/png`：类别分布；
# - `confusion_matrix.png`：混淆矩阵；
# - `per_class_accuracy.csv`：每类准确率；
# - `wrong_samples.csv/png`：错误样例。
#
# 消融结果会自动汇总到项目根目录下的 `ablation_results.csv`，
# 并在当前实验目录额外导出 `ablation_report_table.csv`。
#
# 运行四组实验（终端运行本 .py 文件）：
# - `python leaf_classification_cbam.py --ablation baseline`
# - `python leaf_classification_cbam.py --ablation aug`
# - `python leaf_classification_cbam.py --ablation cbam`
# - `python leaf_classification_cbam.py --ablation cbam_mix`
#
# 报告中的消融实验表格建议如下：
#
# | 方法 | 验证准确率 | 说明 |
# |---|---:|---|
# | ResNet50 baseline | 从 `ablation_results.csv` 填写 | 普通迁移学习 |
# | ResNet50 + 数据增强 | 从 `ablation_results.csv` 填写 | 提升泛化 |
# | ResNet50 + CBAM | 从 `ablation_results.csv` 填写 | 引入注意力机制 |
# | ResNet50 + CBAM + MixUp/CutMix | 从 `ablation_results.csv` 填写 | 最终模型 |
#
# 如果 CBAM 准确率略低也不要慌，可以从错误样例和混淆矩阵分析原因：小数据集下注意力模块可能更容易过拟合，此时可以适当降低学习率、增加数据增强或减少训练轮数。
