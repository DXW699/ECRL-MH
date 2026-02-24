#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_sample_features.py

对单个 (video, audio) 样本抽取多模态微观特征，并保存为 .npz:
- micro_expression : STSTNet 微表情特征
- rppg            : Green 通道 rPPG 特征
- pose            : MediaPipe 姿态特征
- audio_basic     : 基础声学特征
- audio_wav2vec   : wav2vec2 句级 embedding
"""

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
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.abspath(path)


def build_features(
    video_path: str,
    audio_path: str,
    micro_model_path: str = None,
    wav2vec_model_dir: str = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    feats: Dict[str, Any] = {
        "meta_video_path": video_path,
        "meta_audio_path": audio_path,
    }

    # 1) 微表情 (STSTNet)
    if video_path and micro_model_path and \
       os.path.exists(video_path) and os.path.exists(micro_model_path):
        print(f"[build] 微表情特征 (STSTNet) from: {video_path}")
        try:
            micro_ext = MicroExpressionExtractor(
                model_path=micro_model_path,
                device=device,
            )
            micro_vec = micro_ext.extract(video_path)
            feats["micro_expression"] = np.asarray(micro_vec, dtype=np.float32)
            print(f"[build] micro_expression shape: {feats['micro_expression'].shape}")
        except Exception as e:
            print(f"[build] WARNING: 微表情提取失败: {e}")
    else:
        print("[build] NOTE: 缺少视频或微表情权重, 跳过 micro_expression")

    # 2) rPPG
    if video_path and os.path.exists(video_path):
        print(f"[build] rPPG 特征 from: {video_path}")
        try:
            rppg_ext = RPPGExtractor()
            rppg_dict = rppg_ext.extract(video_path)
            # dict -> 向量 (hr, mean, std)
            feats["rppg"] = np.array([
                rppg_dict["heart_rate_bpm"],
                rppg_dict["signal_mean"],
                rppg_dict["signal_std"],
            ], dtype=np.float32)
            print(f"[build] rPPG shape: {feats['rppg'].shape}")
        except Exception as e:
            print(f"[build] WARNING: rPPG 提取失败: {e}")
    else:
        print("[build] NOTE: 缺少视频, 跳过 rPPG")

    # 3) 姿态
    if video_path and os.path.exists(video_path):
        print(f"[build] 姿态特征 from: {video_path}")
        try:
            pose_ext = PoseExtractor()
            pose_vec = pose_ext.extract(video_path)
            feats["pose"] = np.asarray(pose_vec, dtype=np.float32)
            print(f"[build] pose shape: {feats['pose'].shape}")
        except Exception as e:
            print(f"[build] WARNING: pose 提取失败: {e}")
    else:
        print("[build] NOTE: 缺少视频, 跳过 pose")

    # 4) 基础音频
    if audio_path and os.path.exists(audio_path):
        print(f"[build] 基础音频特征 from: {audio_path}")
        try:
            audio_ext = AudioFeatureExtractor()
            audio_dict = audio_ext.extract(audio_path)
            feats["audio_basic"] = np.array(
                [
                    audio_dict["rms_energy"],
                    audio_dict["zero_crossing_rate"],
                    audio_dict["pitch_hz"],
                ],
                dtype=np.float32,
            )
            print(f"[build] audio_basic shape: {feats['audio_basic'].shape}")
        except Exception as e:
            print(f"[build] WARNING: 基础音频特征提取失败: {e}")
    else:
        print("[build] NOTE: 缺少音频, 跳过 audio_basic")

    # 5) wav2vec2
    if audio_path and os.path.exists(audio_path) and wav2vec_model_dir:
        print(f"[build] wav2vec2 特征 from: {audio_path}")
        try:
            wav2vec_vec = extract_wav2vec_features(
                audio_path=audio_path,
                model_dir=wav2vec_model_dir,
                device=device,
                normalize=True,
            )
            feats["audio_wav2vec"] = np.asarray(wav2vec_vec, dtype=np.float32)
            print(f"[build] audio_wav2vec shape: {feats['audio_wav2vec'].shape}")
        except Exception as e:
            print(f"[build] WARNING: wav2vec2 特征提取失败: {e}")
    else:
        if not wav2vec_model_dir:
            print("[build] NOTE: 未提供 wav2vec_model_dir, 跳过 audio_wav2vec")
        else:
            print("[build] NOTE: 缺少音频, 跳过 audio_wav2vec")

    return feats


def main():
    parser = argparse.ArgumentParser(
        description="对单个 (video, audio) 抽取多模态微观特征并保存为 .npz"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="输入视频路径 (如 demo.mp4)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=False,
        default=None,
        help="输入音频路径 (如 demo.wav, 可选)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 .npz 路径 (如 demo_features.npz)",
    )
    parser.add_argument(
        "--micro_model",
        type=str,
        required=False,
        default=None,
        help="STSTNet 微表情权重路径 (如 STSTNet_Weights/006.pth)",
    )
    parser.add_argument(
        "--wav2vec_model_dir",
        type=str,
        required=False,
        default=None,
        help="wav2vec2 模型目录 (含 config.json, pytorch_model.bin)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备: cpu 或 cuda",
    )

    args = parser.parse_args()

    video_path = ensure_abs(args.video)
    audio_path = ensure_abs(args.audio) if args.audio is not None else None
    micro_model_path = ensure_abs(args.micro_model) if args.micro_model is not None else None
    wav2vec_model_dir = ensure_abs(args.wav2vec_model_dir) if args.wav2vec_model_dir is not None else None
    output_path = ensure_abs(args.output)

    print("[build] ==============================================")
    print(f"[build] video_path        : {video_path}")
    print(f"[build] audio_path        : {audio_path}")
    print(f"[build] micro_model_path  : {micro_model_path}")
    print(f"[build] wav2vec_model_dir : {wav2vec_model_dir}")
    print(f"[build] device            : {args.device}")
    print(f"[build] output            : {output_path}")
    print("[build] ==============================================")

    feats = build_features(
        video_path=video_path,
        audio_path=audio_path,
        micro_model_path=micro_model_path,
        wav2vec_model_dir=wav2vec_model_dir,
        device=args.device,
    )

    arrays_to_save = {k: v for k, v in feats.items() if isinstance(v, np.ndarray)}
    if not arrays_to_save:
        raise RuntimeError("没有任何 numpy 特征被抽取, 不保存 .npz")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **arrays_to_save)
    print(f"[build] Saved feature arrays: {list(arrays_to_save.keys())}")
    print(f"[build] Saved to: {output_path}")


if __name__ == "__main__":
    main()
