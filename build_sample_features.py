import argparse
import os
from typing import Dict, Any
import numpy as np
from feature_extraction import (
    MicroExpressionExtractor,
    RPPGExtractor,
    PoseExtractor,
    AudioFeatureExtractor,
)
from audio_wav2vec_features import extract_wav2vec_features

def ensure_abs(path: str) -> str:
    """把相对路径转成绝对路径（以当前工作目录为基准）。"""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.abspath(path)

def build_features(
    video_path: str,
    audio_path: str,
    micro_model_path: str = None,
    rppg_model_path: str = None,
    wav2vec_model_dir: str = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    对一个样本（video + audio）抽取所有可用的特征，并返回一个字典。
    字典中的 numpy 向量会被保存到 .npz 里。
    """
    features: Dict[str, Any] = {
        "meta_video_path": video_path,
        "meta_audio_path": audio_path,
    }

    # -------------------------
    # 1. 微表情特征（视频）
    # -------------------------
    if video_path is not None and micro_model_path is not None:
        if os.path.exists(video_path) and os.path.exists(micro_model_path):
            print(f"[build] Extracting micro-expression features from: {video_path}")
            micro_ext = MicroExpressionExtractor(
                model_path=micro_model_path,
                device=device,
            )
            try:
                micro_feats = micro_ext.extract(video_path)
                features["micro_expression"] = np.asarray(micro_feats, dtype=np.float32)
                print(f"[build] micro_expression shape: {features['micro_expression'].shape}")
            except Exception as e:
                print(f"[build] WARNING: micro-expression extraction failed: {e}")
        else:
            print("[build] WARNING: video or micro_model path not found, skip micro-expression.")
    else:
        print("[build] NOTE: micro_model_path not provided, skip micro-expression features.")

    # -------------------------
    # 2. rPPG 特征（视频 → 心率）
    # -------------------------
    if video_path is not None and os.path.exists(video_path):
        if rppg_model_path:
            print(f"[build] Extracting rPPG features from: {video_path}")
            try:
                rppg_ext = RPPGExtractor(model_path=rppg_model_path, device=device)
                rppg_feats = rppg_ext.extract(video_path)
                features["rppg"] = np.asarray(rppg_feats, dtype=np.float32)
                print(f"[build] rPPG shape: {features['rppg'].shape}")
            except Exception as e:
                print(f"[build] WARNING: rPPG extraction failed: {e}")
        else:
            print("[build] NOTE: rppg_model_path not provided, skip rPPG features.")
    else:
        print("[build] NOTE: video path missing, skip rPPG.")

    # -------------------------
    # 3. 姿态特征（视频 → 骨架）
    # -------------------------
    if video_path is not None and os.path.exists(video_path):
        print(f"[build] Extracting pose features from: {video_path}")
        try:
            pose_ext = PoseExtractor()
            pose_feats = pose_ext.extract(video_path)
            features["pose"] = np.asarray(pose_feats, dtype=np.float32)
            print(f"[build] pose shape: {features['pose'].shape}")
        except ImportError as e:
            print(f"[build] WARNING: PoseExtractor not available (mediapipe not installed?): {e}")
        except Exception as e:
            print(f"[build] WARNING: pose extraction failed: {e}")
    else:
        print("[build] NOTE: video path missing, skip pose features.")

    # -------------------------
    # 4. 传统音频特征（能量/过零率/基频等）
    # -------------------------
    if audio_path is not None and os.path.exists(audio_path):
        print(f"[build] Extracting basic audio features from: {audio_path}")
        try:
            audio_ext = AudioFeatureExtractor()
            audio_feats = audio_ext.extract(audio_path)
            features["audio_basic"] = np.asarray(audio_feats, dtype=np.float32)
            print(f"[build] audio_basic shape: {features['audio_basic'].shape}")
        except Exception as e:
            print(f"[build] WARNING: basic audio feature extraction failed: {e}")
    else:
        print("[build] NOTE: audio path missing, skip basic audio features.")

    # -------------------------
    # 5. wav2vec2 语音 embedding
    # -------------------------
    if audio_path is not None and os.path.exists(audio_path) and wav2vec_model_dir is not None:
        if os.path.isdir(wav2vec_model_dir):
            print(f"[build] Extracting wav2vec2 features from: {audio_path}")
            try:
                wav2vec_feats = extract_wav2vec_features(
                    audio_path=audio_path,
                    model_dir=wav2vec_model_dir,
                    device=device,
                    normalize=True,
                )
                features["audio_wav2vec"] = np.asarray(wav2vec_feats, dtype=np.float32)
                print(f"[build] audio_wav2vec shape: {features['audio_wav2vec'].shape}")
            except Exception as e:
                print(f"[build] WARNING: wav2vec2 feature extraction failed: {e}")
        else:
            print(f"[build] WARNING: wav2vec_model_dir not a directory: {wav2vec_model_dir}")
    else:
        if wav2vec_model_dir is None:
            print("[build] NOTE: wav2vec_model_dir not provided, skip wav2vec2 features.")
        else:
            print("[build] NOTE: audio path missing, skip wav2vec2 features.")

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Build multimodal sample features and save as .npz"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file (e.g., demo.mp4)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=False,
        default=None,
        help="Path to input audio file (e.g., demo.wav). Optional.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .npz file (e.g., demo_features.npz)",
    )
    parser.add_argument(
        "--micro_model",
        type=str,
        default=None,
        help="Path to pre-trained micro-expression model (e.g., STSTNet .pth). Optional.",
    )
    parser.add_argument(
        "--rppg_model",
        type=str,
        default=None,
        help="Path to pre-trained rPPG model (e.g., DeepPhys .pth). Optional.",
    )
    parser.add_argument(
        "--wav2vec_model_dir",
        type=str,
        default=None,
        help="Local directory of wav2vec2 model (contains config.json & pytorch_model.bin). Optional.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference: 'cpu' or 'cuda'",
    )

    args = parser.parse_args()

    video_path = ensure_abs(args.video)
    audio_path = ensure_abs(args.audio) if args.audio is not None else None
    micro_model_path = ensure_abs(args.micro_model) if args.micro_model is not None else None
    rppg_model_path = ensure_abs(args.rppg_model) if args.rppg_model is not None else None
    wav2vec_model_dir = ensure_abs(args.wav2vec_model_dir) if args.wav2vec_model_dir is not None else None
    output_path = ensure_abs(args.output)

    print("[build] ==============================================")
    print(f"[build] video_path        : {video_path}")
    print(f"[build] audio_path        : {audio_path}")
    print(f"[build] micro_model_path  : {micro_model_path}")
    print(f"[build] rppg_model_path   : {rppg_model_path}")
    print(f"[build] wav2vec_model_dir : {wav2vec_model_dir}")
    print(f"[build] device            : {args.device}")
    print(f"[build] output            : {output_path}")
    print("[build] ==============================================")

    feats = build_features(
        video_path=video_path,
        audio_path=audio_path,
        micro_model_path=micro_model_path,
        rppg_model_path=rppg_model_path,
        wav2vec_model_dir=wav2vec_model_dir,
        device=args.device,
    )

    # 只把真正是 numpy 数组的键保存进 .npz
    arrays_to_save = {
        k: v for k, v in feats.items()
        if isinstance(v, np.ndarray)
    }

    if not arrays_to_save:
        raise RuntimeError("No numpy feature arrays extracted; nothing to save.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **arrays_to_save)
    print(f"[build] Saved feature arrays: {list(arrays_to_save.keys())}")
    print(f"[build] Saved to: {output_path}")


if __name__ == "__main__":
    main()
