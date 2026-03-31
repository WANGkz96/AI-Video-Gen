from __future__ import annotations

from pathlib import Path


def probe_video(
    video_path: Path,
    *,
    fallback_width: int,
    fallback_height: int,
    fallback_fps: float,
    fallback_duration: float,
) -> dict[str, float | int]:
    try:
        import cv2  # type: ignore
    except Exception:
        return {
            "width": fallback_width,
            "height": fallback_height,
            "fps": fallback_fps,
            "durationSec": fallback_duration,
        }

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return {
                "width": fallback_width,
                "height": fallback_height,
                "fps": fallback_fps,
                "durationSec": fallback_duration,
            }

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or fallback_width)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or fallback_height)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or fallback_fps)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else fallback_duration

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "durationSec": duration,
        }
    finally:
        capture.release()

