# UI History Archive

本目录保存跨平台改造前的界面版本。所有内容均有 SHA256 记录，未删除历史文件。

## Contents

- `pre_cross_platform_snapshot`: 当前正式 UI 在跨平台改造前的完整快照。
- `pre_cross_platform_snapshot/assets`: 旧版界面演示视频。
- `legacy_original`: 早期双类别原版 UI、对应处理程序和启动脚本。
- `manifest.json`: 原路径、归档路径、文件大小和 SHA256。
- `.gitattributes`: 禁用归档内容的文本换行转换，保证 clone 后 SHA256 不变。

需要在归档内容调整后重建清单时，可在项目根目录运行：

```bash
python archive/ui_history_20260814/archive_ui_history.py --refresh
```

## Restore

需要恢复时，按照 `manifest.json` 中的记录操作。`move` 项可反向移动，`copy` 项可覆盖回原路径。

Archived at: `2026-08-14T11:56:40+08:00`
Files: `31`
Total bytes: `9667309`
