# 使用 bytewandb 时 Lightning WandbLogger 报错的原因与最小修复

本文解释在使用 bytewandb（或非标准 `wandb` 安装）时，训练出现如下报错的原因，并给出最小可行的修复方法。

## 问题现象

- `hydra.errors.InstantiationException: Error in call to target 'lightning.pytorch.loggers.wandb.WandbLogger': ModuleNotFoundError("Requirement 'wandb>=0.12.10' not met …")`

## 根因分析

- Lightning 的 WandbLogger 通过 `lightning_utilities.core.imports.RequirementCache("wandb>=…")` 做依赖检查。
- 该检查依赖发行版元数据（`wandb-*.dist-info`）。若环境是：
  - 使用 fork（如 `bytewandb`，包名不是 `wandb`），或
  - 只有源码模块、缺少 `dist-info`（精简安装/打包导致），
 则虽然 `import wandb` 能成功，但 `RequirementCache` 仍会判定“未满足”。
- 另一个常见问题是入口进程找不到你的补丁（例如 `nequip-train` 使用的 Python 和你运行补丁的 Python 不同）。

## 最小可行修复（四选一）

### 方案一：全局 sitecustomize 补丁（推荐，一次性）

Python 若能在 `sys.path` 上找到 `sitecustomize`，会在启动时自动导入。将下面补丁写入当前环境的 site-packages（一次性安装即可）。

`sitecustomize.py` 内容：

```python
import sys
sys.stderr.write("[bytewandb-patch] sitecustomize loaded\n")
try:
    from lightning_utilities.core.imports import RequirementCache  # type: ignore
    _old = RequirementCache._check_available  # type: ignore[attr-defined]
    def _allow_wandb(self):
        req = getattr(self, "requirement", None)
        if isinstance(req, str) and req.startswith("wandb"):
            self.available = True
            self.message = f"Requirement {req!r} met"
            return
        return _old(self)
    RequirementCache._check_available = _allow_wandb  # type: ignore[assignment]
    sys.stderr.write("[bytewandb-patch] RequirementCache patched\n")
except Exception as e:
    sys.stderr.write(f"[bytewandb-patch] patch skipped: {e!r}\n")
```

一键写入当前环境（示例）：

```bash
python - <<'PY'
import sysconfig, os
content = r'''
import sys
sys.stderr.write("[bytewandb-patch] sitecustomize loaded\n")
try:
    from lightning_utilities.core.imports import RequirementCache  # type: ignore
    _old = RequirementCache._check_available  # type: ignore[attr-defined]
    def _allow_wandb(self):
        req = getattr(self, "requirement", None)
        if isinstance(req, str) and req.startswith("wandb"):
            self.available = True
            self.message = f"Requirement {req!r} met"
            return
        return _old(self)
    RequirementCache._check_available = _allow_wandb  # type: ignore[assignment]
    sys.stderr.write("[bytewandb-patch] RequirementCache patched\n")
except Exception as e:
    sys.stderr.write(f"[bytewandb-patch] patch skipped: {e!r}\n")
'''
dst = os.path.join(sysconfig.get_paths()['purelib'], 'sitecustomize.py')
with open(dst, 'w', encoding='utf-8') as f:
    f.write(content.lstrip())
print('installed:', dst)
PY
```

验证：

```bash
python - <<'PY'
from lightning_utilities.core.imports import RequirementCache
print('RC(wandb>=0.12.10):', bool(RequirementCache('wandb>=0.12.10')))
from lightning.pytorch.loggers import WandbLogger
WandbLogger(project='probe', mode='offline')
print('WandbLogger: constructed OK')
PY
```

之后可直接运行 `nequip-train`，无需额外环境变量。

### 方案二：按次运行的临时补丁（无需安装）

- 将上面的 `sitecustomize.py` 放在任意临时目录，然后带上 `PYTHONPATH` 运行：

```bash
PYTHONPATH="/path/to/dir:${PYTHONPATH}" nequip-train -cn tutorial
```

- 启动时应在 stderr 看到 `[bytewandb-patch] sitecustomize loaded` 与 `RequirementCache patched`。

### 方案三：仅限 nequip-train 的入口补丁

- 在 `nequip/scripts/train.py` 顶部加入相同补丁。适用于始终从 `nequip-train` 进入的场景；若以编程方式调用、未走该入口，则补丁不会执行。

### 方案四（长期稳妥）：制作名为 `wandb` 的 shim 包

- 打一个最小 wheel，包名就叫 `wandb`，版本号满足 `>=0.12.10`，内部 API 直接转发到 `bytewandb`，并提供 `dist-info`。

`pyproject.toml` 示例：

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "wandb"
version = "0.13.87"
requires-python = ">=3.8"
dependencies = ["bytewandb"]
```

`src/wandb/__init__.py`：

```python
from bytewandb import *  # noqa
from bytewandb import __version__  # noqa
```

- 安装该 wheel 后，Lightning 的依赖检查会自然通过，无需补丁。

## 快速绕过（暂不需要 W&B 时）

- 关闭 logger 或改用 CSVLogger：

```bash
nequip-train -cn tutorial trainer.logger=null
# 或
nequip-train -cn tutorial trainer.logger._target_=lightning.pytorch.loggers.CSVLogger
```

## 排查要点

- 确认 `nequip-train` 使用的是预期的 Python（`which nequip-train`，以及脚本 shebang）。
- 若使用临时补丁，务必携带 `PYTHONPATH`；若使用全局补丁，应在进程启动时看到标记。
- Mac 上若使用 MPS，请保持模型 dtype 为 `float32`（MPS 不支持 `float64`）。
