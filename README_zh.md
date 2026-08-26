# Watermark Slayer

**基于 Florence-2 与 LaMA 模型的 AI 智能去水印工具**

🇨🇳 中文 | [🇬🇧 English](README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## 项目概述

`Watermark Slayer` 是一款利用 AI 模型实现精准水印检测与自然去除的应用程序，适用于处理 Sora、Sora 2、Runway 等平台生成的视频和图像。

项目使用 Microsoft Florence-2 识别水印区域，并通过 LaMA 图像修复模型自然填充去除后的区域。软件提供基于 PyWebview 构建的现代化图形界面，操作直观，使用方便。

## 界面截图

![应用界面截图](assets/screenshot-preview.png)

## 效果演示

https://github.com/user-attachments/assets/5b22f737-b0b9-4a82-92d5-0828e4b3a2ff

---

## 功能特性

- **智能检测** - 使用 Florence-2 自动检测水印区域
- **自然去除** - 通过 LaMA 图像修复获得自然的填充效果
- **视频处理** - 支持双阶段视频检测并保留原始音频
- **AI 视频支持** - 可处理 Sora、Sora 2、Runway 等平台生成的视频
- **批量处理** - 支持一次处理整个文件夹
- **预览模式** - 正式处理前可预览检测到的水印区域
- **淡入淡出处理** - 可扩展遮罩范围，处理逐渐出现或消失的水印
- **GPU 加速** - 支持 CUDA 加速处理
- **中英双语界面** - 提供简洁的中文和英文操作界面
- **专业主题** - 提供专注的深色与浅色主题

---

## 安装方法

### Windows

安装脚本会自动下载便携式 Python 环境，无需预先安装系统 Python。

```powershell
git clone https://github.com/CBH2028/watermark_slayer.git
cd watermark_slayer
.\setup.ps1
```

安装完成后，双击 `run.bat` 启动应用程序。

### Linux / macOS

系统需要预先安装 Python 3.10 或更高版本。

```bash
git clone https://github.com/CBH2028/watermark_slayer.git
cd watermark_slayer
chmod +x setup.sh
./setup.sh
```

安装完成后，运行 `./run.sh` 启动应用程序。

### 可选依赖：FFmpeg

如需在处理视频时保留原始音频，请安装 FFmpeg：

- **Windows**：从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载，并将其添加到 `PATH`
- **Linux**：运行 `sudo apt install ffmpeg`
- **macOS**：运行 `brew install ffmpeg`

---

## 使用方法

### 图形界面模式

1. 启动应用程序：Windows 使用 `run.bat`，macOS/Linux 使用 `./run.sh`
2. 在右上角选择语言和界面主题
3. 选择单文件或批量处理模式
4. 设置输入路径和输出路径
5. 根据需要调整处理参数
6. 点击 **开始处理**

应用程序会自动保存当前设置，并在下次启动时恢复。

### 命令行模式

```bash
# 基本用法
python watermark_slayer.py input.png output_folder/

# 使用附加参数
python watermark_slayer.py ./images ./output --overwrite --max-bbox-percent=15 --force-format=PNG

# 使用双阶段检测处理视频
python watermark_slayer.py video.mp4 ./output --detection-skip=3 --fade-in=0.5 --fade-out=0.5

# 预览模式，只检测而不执行去水印
python watermark_slayer.py input.png --preview
```

### 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--overwrite` | 覆盖已经存在的文件 |
| `--transparent` | 将水印区域设为透明，仅适用于图像 |
| `--max-bbox-percent` | 检测框占图像面积的最大百分比，默认值为 100 |
| `--force-format` | 强制指定输出格式：PNG、WEBP、JPG、MP4 或 AVI |
| `--detection-prompt` | 自定义检测提示词，默认值为 `watermark` |
| `--detection-skip` | 处理视频时每隔 N 帧执行一次检测，范围为 1 至 10，默认值为 1 |
| `--fade-in` | 将遮罩向前扩展 N 秒，用于处理淡入水印 |
| `--fade-out` | 将遮罩向后扩展 N 秒，用于处理淡出水印 |
| `--preview` | 只预览检测到的水印，不执行实际处理 |

---

## 视频处理

- **支持格式**：MP4、AVI、MOV、MKV、FLV、WMV、WEBM
- **保留音频**：需要安装 FFmpeg
- **双阶段模式**：将 `--detection-skip` 设置为大于 1 的数值可提高处理速度
- **淡入淡出处理**：使用 `--fade-in` 和 `--fade-out` 处理逐渐出现或消失的水印

---

## 技术栈

- **Florence-2** - Microsoft 视觉模型，用于检测水印
- **LaMA** - 大型遮罩图像修复模型
- **PyWebview** - 跨平台 WebView 图形界面框架
- **Vanilla JavaScript** - 轻量级本地界面逻辑
- **PyTorch** - 深度学习运行后端

---

## 参与贡献

欢迎为项目贡献代码：

1. Fork 本仓库
2. 创建功能分支
3. 提交 Pull Request

---

## 开源协议

本项目采用 MIT License，详细内容请参阅 [LICENSE](LICENSE) 文件。

---

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=CBH2028/watermark_slayer&type=date&legend=top-left)](https://www.star-history.com/#CBH2028/watermark_slayer&type=date&legend=top-left)
