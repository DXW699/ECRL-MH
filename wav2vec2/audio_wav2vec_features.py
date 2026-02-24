import argparse
import os
from typing import Tuple

import numpy as np
import soundfile as sf
import torch

try:
    import librosa
except ImportError:
    librosa = None

from transformers import Wav2Vec2Processor, Wav2Vec2Model


def load_audio_mono_16k(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    加载音频为 mono + 16kHz 采样率，返回 float32 numpy 数组。
    如果原始采样率不是 16k，则使用 librosa 重采样。
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    wav, sr = sf.read(audio_path)

    # 多通道转单通道
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    if sr != target_sr:
        if librosa is None:
            raise RuntimeError(
                f"Audio sample rate is {sr}, but target_sr={target_sr}. "
                f"librosa is not installed, please install it or provide 16kHz audio."
            )
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    wav = wav.astype(np.float32)
    return wav


def load_wav2vec2(model_dir: str, device: str = "cpu") -> Tuple[Wav2Vec2Processor, Wav2Vec2Model]:
    """
    从本地目录加载 wav2vec2 Processor + Model。

    要求 model_dir 下面至少包含:
      - config.json
      - preprocessor_config.json
      - pytorch_model.bin

    如果你现在只有 pytorch_model.bin，请先从对应的官方模型
    (例如 facebook/wav2vec2-base) 下载完整权重目录，然后把这些文件复制过来。
    """
    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f"model_dir is not a directory: {model_dir}")

    print(f"[Wav2Vec2] Loading processor and model from: {model_dir}")
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2Model.from_pretrained(model_dir)

    model.to(device)
    model.eval()
    return processor, model


def extract_wav2vec_features(
    audio_path: str,
    model_dir: str,
    device: str = "cpu",
    normalize: bool = False,
) -> np.ndarray:
    """
    使用本地 wav2vec2 模型对整段音频抽取一个全局 embedding 向量。

    流程:
      1) 加载音频 -> 单声道 16kHz
      2) processor 做特征化
      3) wav2vec2 前向得到 last_hidden_state [B, T, D]
      4) 对时间维做 mean-pooling -> [D] 向量

    参数:
      audio_path: 输入音频路径 (建议 16kHz mono .wav)
      model_dir:  本地 wav2vec2 模型目录
      device:     "cpu" 或 "cuda"
      normalize:  是否对输出向量做 L2 归一化

    返回:
      features: numpy.ndarray, shape = (hidden_size,)
    """
    # 1. 加载音频
    audio = load_audio_mono_16k(audio_path, target_sr=16000)

    # 2. 加载模型
    processor, model = load_wav2vec2(model_dir, device=device)

    # 3. 处理成模型输入
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs["input_values"].to(device)

    # 4. 前向计算
    with torch.no_grad():
        outputs = model(input_values)
        last_hidden_state = outputs.last_hidden_state  # [B, T, D]

    # 5. 时间维平均 -> [D]
    features = last_hidden_state.mean(dim=1).cpu().numpy()[0]  # shape: (D,)

    if normalize:
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract wav2vec2 audio embeddings from a WAV file."
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file (e.g., demo.wav, preferably 16 kHz mono).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Local directory of wav2vec2 model (contains config.json & pytorch_model.bin).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save features as .npy (e.g., features/demo_wav2vec.npy).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, L2-normalize the output feature vector.",
    )

    args = parser.parse_args()

    print(f"[Wav2Vec2] Audio path : {args.audio}")
    print(f"[Wav2Vec2] Model dir  : {args.model_dir}")
    print(f"[Wav2Vec2] Device     : {args.device}")

    feats = extract_wav2vec_features(
        audio_path=args.audio,
        model_dir=args.model_dir,
        device=args.device,
        normalize=args.normalize,
    )

    print(f"[Wav2Vec2] Feature shape: {feats.shape}")
    print(f"[Wav2Vec2] First 10 dims: {feats[:10]}")

    if args.output is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        np.save(args.output, feats.astype(np.float32))
        print(f"[Wav2Vec2] Features saved to: {args.output}")


if __name__ == "__main__":
    main()
