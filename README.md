# CV 课程大作业：ResNet50 + CBAM 叶片分类

本项目已切换为**叶片图像分类**任务，默认使用 PlantVillage 数据集。

## 1. 项目内容

- `leaf_classification_cbam.py`：主训练脚本（支持 4 组消融）
- `prepare_dataset.py`：自动下载并构建 `processed_data`
- `generate_deliverables.py`：自动生成实验报告与汇报 PPT
- `make_submission_zip.py`：按课程命名规范打包提交文件

## 2. 环境安装

```bash
python -m pip install -r requirements.txt
```

## 3. 数据集准备（默认叶片数据）

默认数据源：PlantVillage（GitHub 公开仓库）

```bash
python prepare_dataset.py --source plantvillage --train_per_class 120 --eval_per_class 30 --num_classes 5 --force
```

可选：指定类别（逗号分隔）

```bash
python prepare_dataset.py --source plantvillage --class_names "Tomato___healthy,Potato___Late_blight,Grape___Black_rot,Apple___healthy,Corn_(maize)___Northern_Leaf_Blight" --train_per_class 120 --eval_per_class 30 --force
```

运行后会生成：

- `processed_data/train.txt`
- `processed_data/eval.txt`
- `processed_data/label_list.txt`
- `processed_data/dataset_meta.txt`

> `leaf_classification_cbam.py` 也支持自动准备数据；缺少 `processed_data` 时会自动调用 `prepare_dataset.py`。

## 4. 运行 4 组消融实验

```bash
python leaf_classification_cbam.py --ablation baseline
python leaf_classification_cbam.py --ablation aug
python leaf_classification_cbam.py --ablation cbam
python leaf_classification_cbam.py --ablation cbam_mix
```

快速 smoke test（CPU）：

```bash
python leaf_classification_cbam.py --ablation baseline --warmup_epochs 0 --finetune_epochs 1 --batch_size 16 --train_per_class 20 --eval_per_class 5
```

实验结果会自动汇总到 `ablation_results.csv`。

## 5. 为什么有时 baseline 反而更好？

这通常**不是代码错误**，常见原因是：

1. 训练预算太小：1-2 个 epoch 下，CBAM / MixUp / CutMix 往往还没收敛。
2. 增强强度偏大：小样本或 CPU 快速实验时，强增强会让优化更不稳定。
3. 超参未分组调优：不同方法需要不同学习率、warmup、epoch 才公平。

建议在 GPU 上增加 epoch（例如 warmup 3 + finetune 20 或更高）再比较。

## 6. 自动生成提交材料

```bash
python generate_deliverables.py
```

生成文件：

- `deliverables/report/实验报告_叶片分类_CBAM.docx`
- `deliverables/report/实验报告_简版.md`
- `deliverables/presentation/课程大作业汇报_叶片分类_CBAM.pptx`
- `deliverables/提交材料清单.md`

## 7. 打包提交

```bash
python make_submission_zip.py --group_no 03 --leader 张三 --member1 李四 --member2 王五
```

输出命名示例：`03_张三_李四_王五.zip`

## 8. 参考链接

- PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset
- PlantVillage ZIP: https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip
- ResNet: https://arxiv.org/abs/1512.03385
- CBAM: https://arxiv.org/abs/1807.06521
- MixUp: https://arxiv.org/abs/1710.09412
- CutMix: https://arxiv.org/abs/1905.04899
