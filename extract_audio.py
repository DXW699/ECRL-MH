import sys
sys.path.append("C:/Users/ROG/miniconda3/envs/humanomni/Lib/site-packages/moviepy/video/VideoClip.py")
import argparse
import os

from moviepy.editor import VideoFileClip


def extract_audio(video_path: str, output_wav: str, sr: int = 16000):
    """
    从视频中提取音频为 WAV，单独保存。

    参数:
      video_path: 输入视频路径 (mp4 等)
      output_wav: 输出 WAV 文件路径
      sr:         目标采样率 (默认 16000)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)

    print(f"[extract_audio] Loading video: {video_path}")
    clip = VideoFileClip(video_path)

    if clip.audio is None:
        raise RuntimeError("This video has no audio track.")

    print(f"[extract_audio] Writing audio to: {output_wav}")
    # codec 设置为 pcm_s16le，确保保存为无压缩 16-bit PCM
    clip.audio.write_audiofile(
        output_wav,
        fps=sr,
        nbytes=2,
        codec="pcm_s16le",
        verbose=True,
        logger=None,
    )
    clip.close()
    print("[extract_audio] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract WAV audio from a video file using moviepy."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file, e.g., data/demo.mp4",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output WAV file, e.g., data/demo.wav",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000 Hz).",
    )

    args = parser.parse_args()

    extract_audio(
        video_path=args.video,
        output_wav=args.output,
        sr=args.sr,
    )


if __name__ == "__main__":
    main()
