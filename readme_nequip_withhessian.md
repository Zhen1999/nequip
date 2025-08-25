# NequIP + Full Hessian: 数据转换与训练使用说明

本文说明本仓库为支持“能量/力 + 全 Hessian（EFH）”训练所做的改动，以及从 LMDB 转换到 EXTXYZ、再到分桶+批内补零训练的完整流程与示例。

## 关键改动概览

- 转换器（LMDB -> EXTXYZ）
  - 默认写入“可变宽” per-atom Hessian：每帧 `hessian:R:9*N`，不再全局补到 n_max。
  - 显式开关：
    - `--pad-hessian` 强制补到 `9*n_max`（为了老流程或特殊需要）。
  - 写 EXTXYZ 时强制使用“.extxyz”后缀，避免被当作纯“.xyz”。
- 训练数据管线
  - 新增变换 `RegisterKeys`：将自定义键 `hessian` 注册为 node 字段（在 worker 进程中也会重复注册，幂等）。
  - 新增可 picklable collate：`HessianPadCollate`，在 collate 时按本批最大 N 右侧补零（“只补一次”）。
  - 新增按原子数分桶采样器：`BucketByNumAtomsBatchSampler`，减少无效填充、提升吞吐。
  - DataModule 支持注入 `batch_sampler`/`sampler`（自动传入 dataset，避免 DataLoader 冲突参数）。
- EFH 训练模块
  - `EFHFullHessianLightningModule` 支持监督/正则全 Hessian，兼容目标形状 `(D,D)`、`(N,3,N,3)`、扁平 `D*D`，以及 per-atom 展开 `(N, 9*K)`（K 为补零宽度）。
  - 支持 batch_size>1：训练步内逐帧构建 Hessian 并平均损失。
- 清理
  - 移除将 `hessian` 作为 node 字段的全局默认注册，统一通过 `RegisterKeys` 声明。

## 数据转换（LMDB -> EXTXYZ）

推荐默认（可变宽 + 只在训练时补一次）：

```bash
# 示例：将 val.lmdb 转到 val.extxyz，嵌入 per-atom Hessian（可变宽，不补零）
python -m nequip.misc.lmdb_dataset_conversion.lmdb_to_xyz \
  --in archive/ts1x-val.lmdb \
  --out archive/ts1x-val.extxyz \
  --pos pos --charges charges --forces forces --energy energy \
  --hessian hessian --embed-hessian
```

如需老式“补到 n_max”：添加 `--pad-hessian`。

### 快速子集转换脚本（部分转换）

我们提供了简单脚本用于抽样/限量转换：`misc/lmdb_dataset_conversion/convert_subset.sh`

用法：

```bash
bash misc/lmdb_dataset_conversion/convert_subset.sh \
  -i archive/ts1x_hess_train_big.lmdb \
  -o archive/ts1x_hess_train_big.sub200.extxyz \
  -l 200 -k 50
```

- `-i`：输入 LMDB 路径
- `-o`：输出 EXTXYZ 路径（脚本也接受“.xyz”，转换器会自动转为“.extxyz”）
- `-l`：最多转换 N 条（可选）
- `-k`：抽样步长 every=k（可选）
- `--pad`：将可变宽改为全局补零（可选）

脚本默认键位：`pos/charges/forces/energy/hessian`；如你的数据键不同，请直接用 Python 命令行自行指定对应 `--pos/--charges/...`。

## 训练配置（分桶 + 批内补零）

我们已提供示例配置：`configs/ts1x_xyz_subset.yaml`，要点如下：

- ASEDataset 读取 `.extxyz`
- transforms：
  - `ChemicalSpeciesToAtomTypeMapper`
  - `NeighborListTransform`
  - `RegisterKeys(node_fields: [hessian])`
- include_keys: `[hessian]`
- DataLoader：
  - collate_fn: `nequip.data.datamodule._collate.hessian_pad_collate_factory`（批内补零）
  - batch_sampler: `nequip.data.BucketByNumAtomsBatchSampler`（按 N 分桶）
- 训练模块：`nequip.train.EFHFullHessianLightningModule`（EFH）

运行：

```bash
# 在 configs 目录下
nequip-train -cn ts1x_xyz_subset
```

可调参数建议：
- batch_size：根据内存/显存增大
- bucket_width：8 或 16（N 分布更集中时效果更好）
- num_workers：4–8
- pin_memory: true（可选）

## 端到端示例

1) 子集转换（可变宽 + 嵌入 Hessian）：

```bash
python -m nequip.misc.lmdb_dataset_conversion.lmdb_to_xyz \
  --in archive/ts1x_hess_train_big.lmdb \
  --out archive/ts1x_hess_train_big.sub200.extxyz \
  --pos pos --charges charges --forces forces --energy energy \
  --hessian hessian --embed-hessian --limit 200 --every 50
```

2) 配置指向 `.extxyz`，并使用分桶 + 批内补零（见 `configs/ts1x_xyz_subset.yaml`）

3) 训练/验证：

```bash
nequip-train -cn ts1x_xyz_subset
```

## 故障排查

- “Unregistered key hessian”：确认 transforms 中包含 `RegisterKeys(node_fields: [hessian])`；我们已在变换 `__call__` 中重复注册以兼容多进程。
- MPS/GPU 精度：Hessian 常用 float64，macOS MPS 对 float64 支持有限；建议 CPU 或确保 dtype 兼容。
- EXTXYZ 后缀：嵌入属性的文件请使用“.extxyz”，避免被当作纯“.xyz”。

## 变更清单（文件级）

- 转换器：`misc/lmdb_dataset_conversion/lmdb_to_xyz.py`
  - 默认可变宽，`--pad-hessian` 显式补零；写入 extxyz 强制“.extxyz”。
- 变换：`nequip/data/transforms/register_extra.py`（新增 + 导出）
  - `RegisterKeys` 支持在 worker 中重复注册。
- collate：`nequip/data/datamodule/_collate.py`（新增）
  - `HessianPadCollate` 与工厂 `hessian_pad_collate_factory`。
- 分桶采样器：`nequip/data/_bucket_sampler.py`（新增）并在 `nequip/data/__init__.py` 导出。
- DataModule：`nequip/data/datamodule/_base_datamodule.py`
  - 支持注入 `batch_sampler`/`sampler` 并自动传入 dataset。
- EFH 模块：`nequip/train/efh_full_module.py`（新增）。
- 示例配置：`configs/ts1x_xyz_subset.yaml`（更新，指向 .extxyz + 分桶 + 批内补零）。
- 清理：`nequip/data/_key_registry.py` 移除对 `hessian` 的全局默认注册。

---

如需我再生成大规模数据的批量转换脚本或提供一份“生产配置”模板，请告知。
