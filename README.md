# CV 课程大作业：ResNet50 + CBAM 花卉分类

本项目已按《项目要求.pptx》补齐可提交材料，并提供可直接运行的训练与消融流程。

## 1. 项目内容

- `leaf_classification_cbam.py`：主训练脚本（支持 4 组消融）
- `prepare_dataset.py`：一键下载并构建 `processed_data`
- `generate_deliverables.py`：自动生成实验报告与汇报 PPT
- `requirements.txt`：依赖清单
- `datasets/`：公开数据集原始文件
- `processed_data/`：训练可直接读取的数据格式
- `outputs_leaf_ablation/`：每组实验输出
- `deliverables/`：最终提交材料

## 2. 环境安装

```bash
python -m pip install -r requirements.txt
```

## 3. 数据集准备

项目默认使用 TensorFlow Flowers（5 类公开数据集）：

```bash
python prepare_dataset.py --train_per_class 120 --eval_per_class 30 --force
```

运行后会生成：

- `processed_data/train.txt`
- `processed_data/eval.txt`
- `processed_data/label_list.txt`

> 备注：`leaf_classification_cbam.py` 也支持自动下载与准备数据，不手动执行本步骤也可运行。

## 4. 运行 4 组消融实验

```bash
python leaf_classification_cbam.py --ablation baseline --warmup_epochs 0 --finetune_epochs 1 --batch_size 16
python leaf_classification_cbam.py --ablation aug --warmup_epochs 0 --finetune_epochs 1 --batch_size 16
python leaf_classification_cbam.py --ablation cbam --warmup_epochs 0 --finetune_epochs 1 --batch_size 16
python leaf_classification_cbam.py --ablation cbam_mix --warmup_epochs 0 --finetune_epochs 1 --batch_size 16
```

实验结果会自动汇总到 `ablation_results.csv`。

## 5. 当前已跑结果（本机，CPU，快速设置）

| 模型 | 验证准确率 | 说明 |
|---|---:|---|
| ResNet50 baseline | 88.67% | 普通迁移学习 |
| ResNet50 + 数据增强 | 88.00% | 提升泛化 |
| ResNet50 + CBAM | 88.00% | 引入注意力机制 |
| ResNet50 + CBAM + MixUp/CutMix | 84.00% | 最终模型 |

> 说明：以上为 `finetune_epochs=1` 的快速 smoke test 结果；正式报告建议增加 epoch 并优先使用 GPU。

## 6. 自动生成提交材料

```bash
python generate_deliverables.py
```

生成文件：

- `deliverables/report/实验报告_叶片分类_CBAM.docx`
- `deliverables/report/实验报告_简版.md`
- `deliverables/presentation/课程大作业汇报_叶片分类_CBAM.pptx`
- `deliverables/提交材料清单.md`

## 7. 提交打包建议

根据课程要求，最终将以下内容打包：

1. 实验报告（Word 或 PDF）
2. 源码（`.py`/`.ipynb`）
3. 数据集（`processed_data/`，可附 `datasets/`）
4. 汇报 PPT

命名格式示例：`03_组长姓名_组员1姓名_组员2姓名.zip`

可使用脚本自动打包：

```bash
python make_submission_zip.py --group_no 03 --leader 张三 --member1 李四 --member2 王五
```

## 8. 数据与参考来源

- TF Flowers（数据集页面）：https://www.tensorflow.org/datasets/catalog/tf_flowers
- TF Flowers（下载地址）：https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
- ResNet 论文：https://arxiv.org/abs/1512.03385
- CBAM 论文：https://arxiv.org/abs/1807.06521
