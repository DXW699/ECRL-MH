import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
from torch import nn

# Optional imports for rPPG and pose extraction
try:
    from scipy.signal import butter, filtfilt, find_peaks
    from scipy.io import wavfile
except ImportError:
    butter = filtfilt = find_peaks = wavfile = None

# Optional import for mediapipe pose extraction
try:
    import mediapipe as mp
except ImportError:
    mp = None


@dataclass
class MicroExpressionExtractor:
    """
    Extract micro-expression features using STSTNet pre-trained model.
    """
    model_path: str
    device: str = "cpu"
    input_size: int = 224
    flow_step: int = 2  # skip frames to reduce computation

    def __post_init__(self):
        if torch is None:
            raise ImportError(
                "PyTorch is required for MicroExpressionExtractor but is not installed. "
                "Please install torch if you want to use a deep micro-expression model."
            )

        self.device = torch.device(self.device)
        self.model: Optional[nn.Module] = None

        if self.model_path and os.path.exists(self.model_path):
            try:
                # Attempt to load pre-trained model
                ckpt = torch.load(self.model_path, map_location=self.device)
                if isinstance(ckpt, nn.Module):
                    self.model = ckpt.to(self.device)
                    self.model.eval()
                    print(f"[MicroExp] Loaded full micro-expression model from {self.model_path}")
                else:
                    print(
                        "[MicroExp][WARN] Model checkpoint is state_dict, fallback to optical flow features."
                    )
                    self.model = None
            except Exception as e:
                print(
                    f"[MicroExp][WARN] Failed to load model checkpoint ({e}); fallback to optical flow."
                )
                self.model = None
        else:
            print("[MicroExp] No model path provided; using optical flow.")

    def _iter_flow(self, video_path: str) -> List[np.ndarray]:
        """Compute dense optical flow for every `flow_step` frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Failed to read first frame from video")
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        flows: List[np.ndarray] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % self.flow_step != 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            flows.append(flow)
            prev_gray = gray

        cap.release()
        return flows

    def _preprocess_flow(self, flow: np.ndarray) -> np.ndarray:
        """Convert optical flow to a 3-channel uint8 image suitable for CNN input."""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255  # full saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr_resized = cv2.resize(bgr, (self.input_size, self.input_size))
        bgr_resized = bgr_resized.astype(np.float32) / 255.0  # [0, 1]
        return bgr_resized

    def _handcrafted_stats(self, flows: List[np.ndarray]) -> np.ndarray:
        """
        Fallback: compute simple statistics on optical flow magnitude / components
        across time, yielding a small fixed-length feature vector.
        """
        per_frame_stats: List[np.ndarray] = []
        for flow in flows:
            u = flow[..., 0]
            v = flow[..., 1]
            mag, _ = cv2.cartToPolar(u, v)
            per_frame_stats.append(
                np.array(
                    [
                        float(mag.mean()),
                        float(mag.std()),
                        float(mag.max()),
                        float(np.abs(u).mean()),
                        float(np.abs(v).mean()),
                    ],
                    dtype=np.float32,
                )
            )
        feature_matrix = np.vstack(per_frame_stats)  # [T, 5]
        aggregated = feature_matrix.mean(axis=0)     # [5]
        return aggregated

    def extract(self, video_path: str) -> np.ndarray:
        """
        Extract micro-expression features from a video.

        If a valid deep model is loaded, run it on flow images per frame and
        average features. Otherwise, return handcrafted optical-flow statistics.
        """
        flows = self._iter_flow(video_path)
        if not flows:
            raise ValueError("No frames processed for optical flow features")

        # Case 1: we have a deep model -> use it
        if self.model is not None:
            features: List[np.ndarray] = []
            for flow in flows:
                flow_img = self._preprocess_flow(flow)  # (H, W, 3)
                input_tensor = torch.from_numpy(flow_img.transpose(2, 0, 1)).unsqueeze(0)
                input_tensor = input_tensor.to(self.device)
                with torch.no_grad():
                    output = self.model(input_tensor)
                features.append(output.cpu().numpy().flatten())
            feature_matrix = np.vstack(features)
            aggregated = feature_matrix.mean(axis=0)
            return aggregated

        # Case 2: no deep model -> handcrafted stats
        return self._handcrafted_stats(flows)


@dataclass
class RPPGExtractor:
    """Extract remote photoplethysmography (rPPG) signals and heart rate."""

    model_path: str
    fps: Optional[float] = None
    device: str = "cpu"

    def __post_init__(self):
        if os.path.exists(self.model_path):
            try:
                # Assuming the model is DeepPhys or similar for rPPG extraction
                self.model = torch.load(self.model_path, map_location=self.device)
                self.model.eval()  # Set to evaluation mode
                print(f"[RPPG] Loaded pre-trained rPPG model from {self.model_path}")
            except Exception as e:
                print(f"[RPPG] Failed to load model ({e}); fallback to simple rPPG extraction.")
                self.model = None
        else:
            self.model = None
            print("[RPPG] Model path does not exist; fallback to simple rPPG extraction.")

    def extract(self, video_path: str) -> np.ndarray:
        """Extract rPPG features."""
        if self.model is not None:
            return self.extract_using_model(video_path)
        else:
            return self.extract_simple_rppg(video_path)

    def extract_using_model(self, video_path: str) -> np.ndarray:
        """Extract rPPG features using a deep learning model."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            roi = frame[:, :, 1]  # Green channel
            frames.append(roi)
        cap.release()

        frames = np.stack(frames, axis=0)  # [T, H, W]
        frames = torch.tensor(frames).unsqueeze(0).to(self.device)  # [1, T, H, W]

        with torch.no_grad():
            output = self.model(frames)
            return output.cpu().numpy()

    def extract_simple_rppg(self, video_path: str) -> np.ndarray:
        """Fallback: extract rPPG using the green channel."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        g_values: List[float] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            roi = frame[:, :, 1]  # Green channel
            g_values.append(float(np.mean(roi)))
        cap.release()

        g_values = np.array(g_values)
        g_values = g_values - np.mean(g_values)
        return np.array([np.mean(g_values), np.std(g_values)])


@dataclass
class PoseExtractor:
    """Extract simple pose features using MediaPipe's pose estimation."""

    def __post_init__(self):
        if mp is None:
            raise ImportError(
                "MediaPipe is required for pose extraction but is not installed. "
                "Please install mediapipe via pip."
            )
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def extract(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        features: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            if results.pose_landmarks is None:
                continue
            landmarks = results.pose_landmarks.landmark
            # Example feature: shoulder width normalized by torso length
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
            # Compute Euclidean distances
            shoulder_dist = np.linalg.norm([
                left_shoulder.x - right_shoulder.x,
                left_shoulder.y - right_shoulder.y
            ])
            hip_dist = np.linalg.norm([
                left_hip.x - right_hip.x,
                left_hip.y - right_hip.y
            ])
            # Normalized shoulder width (bigger value may indicate expansive posture)
            norm_shoulder_width = shoulder_dist / (hip_dist + 1e-6)
            # Example feature: arm openness (distance between wrists and shoulders)
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            arm_openness = (
                np.linalg.norm([left_wrist.x - left_shoulder.x, left_wrist.y - left_shoulder.y]) +
                np.linalg.norm([right_wrist.x - right_shoulder.x, right_wrist.y - right_shoulder.y])
            )
            features.append(np.array([norm_shoulder_width, arm_openness], dtype=np.float32))
        cap.release()
        if not features:
            raise ValueError("No pose landmarks detected in video")
        feature_matrix = np.vstack(features)
        # Aggregate features by median to reduce the influence of outliers
        aggregated = np.median(feature_matrix, axis=0)
        return aggregated


@dataclass
class AudioFeatureExtractor:
    """Extract simple acoustic features from a WAV audio file."""

    sample_rate: Optional[int] = None

    def extract(self, audio_path: str) -> Dict[str, Any]:
        if wavfile is None or find_peaks is None:
            raise ImportError("scipy.io.wavfile and scipy.signal are required for audio feature extraction")
        sr, signal = wavfile.read(audio_path)
        if self.sample_rate is not None and sr != self.sample_rate:
            pass  # Resampling can be added if needed
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32)
        if signal.ndim == 2:
            signal = np.mean(signal, axis=1)
        signal = signal - np.mean(signal)
        energy = float(np.sqrt(np.mean(signal ** 2)))
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        zcr = float(len(zero_crossings) / len(signal))
        pitch_hz = self._estimate_pitch(signal, sr)
        return {
            'rms_energy': energy,
            'zero_crossing_rate': zcr,
            'pitch_hz': pitch_hz
        }

    def _estimate_pitch(self, signal: np.ndarray, sr: int) -> float:
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[len(corr) // 2:]
        d = np.diff(corr)
        try:
            start = np.where(d > 0)[0][0]
        except IndexError:
            return 0.0
        peaks, _ = find_peaks(corr[start:])
        if peaks.size == 0:
            return 0.0
        peak = peaks[0] + start
        period = peak / sr
        if period > 0:
            return 1.0 / period
        return 0.0


def main():
    """Optional demo of feature extraction on a provided video and audio file."""
    import argparse
    parser = argparse.ArgumentParser(description="Extract multimodal features for psychological analysis")
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--audio', type=str, help='Path to input audio file (wav)')
    parser.add_argument('--micro_model', type=str, help='Path to pre-trained micro-expression model')
    parser.add_argument('--device', type=str, default='cpu', help='Device for model inference')
    args = parser.parse_args()
    if args.video and args.micro_model:
        micro_ext = MicroExpressionExtractor(model_path=args.micro_model, device=args.device)
        micro_feats = micro_ext.extract(args.video)
        print('Micro-expression features:', micro_feats)
    if args.video:
        rppg_ext = RPPGExtractor()
        rppg_feats = rppg_ext.extract(args.video)
        print('rPPG features:', rppg_feats)
        try:
            pose_ext = PoseExtractor()
            pose_feats = pose_ext.extract(args.video)
            print('Pose features:', pose_feats)
        except ImportError as e:
            print('Pose extraction skipped:', e)
    if args.audio:
        audio_ext = AudioFeatureExtractor()
        audio_feats = audio_ext.extract(args.audio)
        print('Audio features:', audio_feats)


if __name__ == '__main__':
    main()
