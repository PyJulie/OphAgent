# 运行资产与完整安装

OphAgent 源码仓库不包含模型 checkpoint。完整的本地配置由公开源码和单独
分发的运行资产压缩包共同组成：

`OphAgent-runtime-assets-0.1.0.zip`

压缩包名称、大小、SHA-256、目录结构和资产数量记录在
[`reviewer/assets.json`](../reviewer/assets.json) 中。请通过对应版本的授权
分发渠道获取压缩包；下载地址不写入公开仓库。

## 分发边界

| 位置 | 内容 |
|---|---|
| 源码仓库 | 源码、测试、配置模板、组件清单、checkpoint 元数据和资产安装器 |
| 运行资产压缩包 | `manuscript-full` 配置所需的 37 个 checkpoint 和资产清单 |
| 本地运行目录 | API 配置、外部组件、获准使用的输入、会话、报告、日志和缓存 |

运行目录应位于 Git 仓库之外。运行资产压缩包不包含 API 密钥、网页密码、
数据集、临床输入、会话、日志或已生成报告。

## 环境要求

- Python 3.10 或更高版本，以及 Git。
- 完整运行本地工具时建议使用支持 CUDA 的 NVIDIA GPU。
- 文件系统需支持大于 4 GB 的单个文件。
- 如果压缩包和解压目录位于同一磁盘，建议至少保留 50 GiB 空间，并为
  Python 环境及模型缓存预留额外空间。
- 安装 Python 包、固定版本的外部组件及调用所选模型 API 时需要联网。

## 1. 克隆并安装代码

PowerShell：

```powershell
git clone https://github.com/PyJulie/OphAgent.git
Set-Location .\OphAgent
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[oct-disc]"
```

bash 或 zsh：

```bash
git clone https://github.com/PyJulie/OphAgent.git
cd OphAgent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[oct-disc]"
```

没有 checkpoint 时仍可检查源码和运行离线安全测试；依赖 checkpoint 的
工具不可用属于代码仓库本身的预期状态，不代表完整配置已经运行。

## 2. 校验并安装运行资产

下载后的 ZIP 应保存在 Git 仓库之外。

PowerShell：

```powershell
.\.venv\Scripts\python reviewer\install_assets.py `
  --archive "$HOME\Downloads\OphAgent-runtime-assets-0.1.0.zip" `
  --runtime-dir "$HOME\ophagent-runtime"
$env:OPHAGENT_RUNTIME_DIR = (Resolve-Path "$HOME\ophagent-runtime").Path
```

bash 或 zsh：

```bash
.venv/bin/python reviewer/install_assets.py \
  --archive "$HOME/Downloads/OphAgent-runtime-assets-0.1.0.zip" \
  --runtime-dir "$HOME/ophagent-runtime"
export OPHAGENT_RUNTIME_DIR="$HOME/ophagent-runtime"
```

安装器会核对压缩包大小和 SHA-256、拒绝不安全的 ZIP 路径、确认 37 个
checkpoint 均存在，并将运行资产解压至仓库外。已有运行目录不会被覆盖。

只校验下载文件而不解压：

```bash
python reviewer/install_assets.py \
  --archive /path/to/OphAgent-runtime-assets-0.1.0.zip \
  --verify-only
```

## 3. 安装固定版本的外部组件

ReT-SAM 2.0 和 G-DISC 推理代码已包含在仓库中。其他具有明确许可证的
组件可按锁定版本安装：

```bash
ophagent-components install --all
```

RetiZero 和 FMUE 对应版本未声明许可证，仅在用户明确接受其上游条款后
获取：

```bash
ophagent-components install retizero fmue --allow-unlicensed
```

Windows 未激活虚拟环境时，请使用 `.venv/Scripts/` 下的对应命令。

## 4. 在本地配置模型供应商

资产安装器会根据 `.env.example` 创建
`<OPHAGENT_RUNTIME_DIR>/.env`。在该文件中填写所选供应商的 API 密钥，
不要将密钥放入 Git 仓库。

支持的官方供应商和网关包括 OpenAI、Anthropic Claude、Google Gemini、
DashScope、AIGCBest 和 OpenRouter。网页端也支持在
**Personalize > API** 中配置用户自己的密钥。托管 API 可能由用户自己的
供应商账户产生费用。

## 5. 验证完整配置

按以下顺序运行：

```bash
ophagent-assets verify --profile manuscript-full
ophagent-components status --profile manuscript-full
python -m ophagent.reviewer_smoke
ophagent-preflight --quick --json --no-save-json
ophagent-preflight --json --no-save-json
```

预期结果：

- 权重检查显示 37 个 checkpoint 均通过校验。
- 组件检查显示所需推理组件均为 ready。
- 离线安全测试返回包含 `"ok": true` 的 JSON。
- 仅当所选供应商、checkpoint、组件源码和四种模态均可运行时，完整预检
  才会以退出码 `0` 结束并显示 `strict_ready=true`。

完整预检会加载本地模型组件并探测所配置的 API，因此耗时长于快速预检。
各项检查的边界见 [`VALIDATION.md`](VALIDATION.md)。

## 6. 启动 OphAgent

网页端：

```bash
ophagent-web
```

浏览器打开 `http://127.0.0.1:8765`。

命令行：

```bash
python demos/chat.py
```

复现具体运行时，应记录实际供应商、解析后的模型标识、effort、任务定义、
输入、日期、源码提交和资产配置。托管模型别名可能随时间变化。

