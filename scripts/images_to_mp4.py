#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _natural_sort_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def _collect_images(input_dir: Path) -> List[Path]:
    images = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images, key=_natural_sort_key)


def _resolve_output_path(input_dir: Path, output: Path | None) -> Path:
    default_name = f"{input_dir.name}.mp4"

    if output is None:
        return input_dir.parent / default_name

    output = output.expanduser()

    # If user passed an existing directory, place default filename inside it.
    if output.exists() and output.is_dir():
        return output / default_name

    # If user passed a path with no extension, assume they meant a filename stem.
    if output.suffix == "":
        return output.with_suffix(".mp4")

    return output


def _open_video_writer(
    output_path: Path,
    frame_size: Sequence[int],
    fps: float,
    codec_candidates: Iterable[str],
) -> cv2.VideoWriter:
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(
            f"Output path is a directory, not a file: {output_path}"
        )

    if output_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Output path must have a video extension {sorted(VIDEO_EXTENSIONS)}. "
            f"Got: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(frame_size[0]), int(frame_size[1])

    last_error = None
    for codec in codec_candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            try:
                print(
                    f"Using codec={codec}, backend={writer.getBackendName()}, output={output_path}"
                )
            except Exception:
                print(f"Using codec={codec}, output={output_path}")
            return writer
        writer.release()
        last_error = codec

    codec_str = ", ".join(codec_candidates)
    raise RuntimeError(
        f"Could not open video writer for {output_path}. " f"Tried codecs: {codec_str}"
    )


def images_to_mp4(input_dir: Path, fps: float, output_path: Path, resize: bool) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if fps <= 0:
        raise ValueError(f"FPS must be > 0. Got: {fps}")

    image_paths = _collect_images(input_dir)
    if not image_paths:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"No supported images found in {input_dir}. Supported extensions: {exts}"
        )

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Failed to read first image: {image_paths[0]}")

    height, width = first.shape[:2]

    # Container-aware codec preference
    if output_path.suffix.lower() == ".avi":
        codec_candidates = ("MJPG", "XVID")
    else:
        codec_candidates = ("mp4v", "avc1", "H264")

    writer = _open_video_writer(
        output_path=output_path,
        frame_size=(width, height),
        fps=fps,
        codec_candidates=codec_candidates,
    )

    frames_written = 0
    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            if frame.shape[:2] != (height, width):
                if not resize:
                    raise ValueError(
                        "Image size mismatch. "
                        f"Expected {(height, width)}, got {frame.shape[:2]} for {image_path}. "
                        "Use --resize to auto-resize all frames to the first image size."
                    )
                frame = cv2.resize(
                    frame, (width, height), interpolation=cv2.INTER_LINEAR
                )

            writer.write(frame)
            frames_written += 1
    finally:
        writer.release()

    if frames_written == 0:
        raise RuntimeError("No frames were written to the output video.")

    print(f"Wrote {frames_written} frames to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all images in a folder into an MP4 video."
    )
    parser.add_argument("input_dir", type=Path, help="Folder that contains images.")
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Output video frame rate (e.g. 24, 30).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output video file path, or an output directory. "
            "If a directory is given, the output becomes <dir>/<input_dir_name>.mp4"
        ),
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize mismatched image sizes to the first image size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = _resolve_output_path(input_dir, args.output)

    images_to_mp4(
        input_dir=input_dir,
        fps=float(args.fps),
        output_path=output_path,
        resize=bool(args.resize),
    )


if __name__ == "__main__":
    main()
