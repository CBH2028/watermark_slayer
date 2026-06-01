# Branch Description

Branch: `main`

Latest upload scope: realtime processing preview and UI simplification.

## Update Notes

- `[VIDEO]` Added live per-frame preview events so Florence detection frames and iopaint results can be shown during video processing.
- `[VIDEO]` Added a local read-only media service with range support and frame-based JPEG playback to avoid browser video codec limitations.
- `[UI]` Added a dedicated processing preview modal for before/after images and frame previews.
- `[UI]` Removed the original main-page before/after comparison panel so processing visuals stay focused in the modal.
- `[PROGRESS]` Moved processing progress from the main page into the processing preview modal.
- `[LAYOUT]` Enlarged and expanded the runtime log area for easier reading.
- `[WINDOW]` Set the desktop GUI to open maximized by default.
- `[COMPAT]` Re-encoded MP4 output as H.264/AAC with fast-start metadata when FFmpeg is available.

## Verification

- `[CHECK]` Python compile checks passed for the main runtime and GUI modules.
- `[CHECK]` JSON validation passed for UI language and config files.
- `[CHECK]` Frontend DOM references and Python bridge method references were checked.
- `[NOTE]` Local GUI launch is still blocked on this machine by the missing GTK `gi` module.
