"""Extractor for canonical 3D hand landmarks poses for 3D visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from sinala.data.alphabet_dataset import ALPHABET_CLASSES
from sinala.data.hand_landmark_extractor import HandLandmarkExtractor


class CanonicalPoseExtractor:
    """Computes and exports centered 3D hand poses and sequences for interactive Three.js rendering."""

    CANONICAL_GREETING_VIDEOS: dict[str, str] = {
        "Oi": "acessibilidade_brasil__Oi__source_oi__oi.mp4",
        "Nome": "acessibilidade_brasil__Nome__source_nome__nome.mp4",
        "Obrigado": "acessibilidade_brasil__Obrigado__source_obrigado_1__obrigado1.mp4",
        "Sim": "v_librasil__Sim__Articulador1__Sim_Articulador1.mp4",
        "Não": "v_librasil__Não__Articulador1__Não_Articulador1.mp4",
        "Desculpa": "v_librasil__Desculpa__Articulador1__Desculpa_Articulador1.mp4",
        "Por favor": "v_librasil__Por favor__Articulador1__Por favor_Articulador1.mp4",
        "Ajudar": "v_librasil__Ajudar__Articulador1__Ajudar_Articulador1.mp4",
        "Cumprimento": "v_librasil__Cumprimento__Articulador1__Cumprimento_Articulador1.mp4",
    }

    def __init__(
        self,
        raw_alphabet_dir: str | Path = "data/raw/alphabet",
        greetings_dir: str | Path = "data/landmarks-greetings",
        raw_greetings_video_dir: str | Path = "data/raw/greetings/videos",
        model_asset_path: str | Path = "models/hand_landmarker.task",
        output_dir: str | Path = "web/data",
    ) -> None:
        self.raw_alphabet_dir = Path(raw_alphabet_dir)
        self.greetings_dir = Path(greetings_dir)
        self.raw_greetings_video_dir = Path(raw_greetings_video_dir)
        self.model_asset_path = Path(model_asset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_and_export(self, filename: str = "hand_poses_3d.json") -> Path:
        """Extract canonical 3D poses for each letter A-Z and sequences for greetings."""
        poses: dict[str, Any] = {}

        # 1. Alphabet static poses
        for letter in ALPHABET_CLASSES:
            csv_path = self.raw_alphabet_dir / f"{letter}.csv"
            if not csv_path.exists():
                continue

            data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            landmarks_3d = data.reshape(-1, 21, 3)
            canonical_pose = np.median(landmarks_3d, axis=0)

            wrist = canonical_pose[0].copy()
            centered = canonical_pose - wrist
            scale = float(np.linalg.norm(centered[9]))
            if scale < 1e-6:
                scale = 1.0
            normalized_3d = centered / scale

            # Invert Y and Z for Three.js (palm facing camera, fingers up)
            normalized_3d[:, 1] = -normalized_3d[:, 1]
            normalized_3d[:, 2] = -normalized_3d[:, 2]

            # Center around bounding box center so the hand is dead-center
            bbox_min = normalized_3d.min(axis=0)
            bbox_max = normalized_3d.max(axis=0)
            bbox_center = (bbox_min + bbox_max) / 2.0
            centered_3d = normalized_3d - bbox_center

            points = []
            for pt in centered_3d:
                points.append([round(float(pt[0]), 4), round(float(pt[1]), 4), round(float(pt[2]), 4)])

            poses[letter] = {
                "type": "alphabet",
                "letter": letter,
                "landmarks": points,
            }

        # 2. Relaxed neutral open hand pose for sign transitions
        if "B" in poses:
            open_hand_pts = np.array(poses["B"]["landmarks"], dtype=np.float32).copy()
            open_hand_pts[1, 0] -= 0.08
            open_hand_pts[2, 0] -= 0.16
            open_hand_pts[3, 0] -= 0.22
            open_hand_pts[4, 0] -= 0.28
            open_points = [[round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)] for p in open_hand_pts]
            poses["_OPEN"] = {
                "type": "neutral",
                "landmarks": open_points,
            }

        # 3. Dynamic greetings motion sequences from raw curated videos
        if self.raw_greetings_video_dir.exists() and self.model_asset_path.is_file():
            extractor = HandLandmarkExtractor(self.model_asset_path)
            for label, video_name in self.CANONICAL_GREETING_VIDEOS.items():
                video_path = self.raw_greetings_video_dir / video_name
                if not video_path.exists():
                    continue

                cap = cv2.VideoCapture(str(video_path))
                raw_frames: list[np.ndarray] = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, res = extractor.process(frame)
                    if res.hand_landmarks and len(res.hand_landmarks) > 0:
                        raw_frames.append(np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float32))
                cap.release()

                if not raw_frames:
                    continue

                # Sample 20 to 28 frames uniformly across gesture
                sample_count = min(len(raw_frames), 24)
                indices = np.linspace(0, len(raw_frames) - 1, sample_count, dtype=int)
                sampled = [raw_frames[i] for i in indices]
                # Anchor to apex frame (middle frame)
                mid_idx = len(sampled) // 2
                mid_pts = sampled[mid_idx]
                wrist_mid = mid_pts[0].copy()
                hand_scale = float(np.linalg.norm(mid_pts[9] - wrist_mid)) or 0.2

                # Compute apex shape and its bounding box center
                apex_shape = (mid_pts - wrist_mid) / hand_scale
                apex_shape[:, 1] = -apex_shape[:, 1]
                apex_shape[:, 2] = -apex_shape[:, 2]
                apex_center = (apex_shape.min(axis=0) + apex_shape.max(axis=0)) / 2.0

                frame_list: list[list[list[float]]] = []
                for pts in sampled:
                    wrist = pts[0].copy()
                    # Pure anatomical hand shape relative to its own wrist
                    shape = (pts - wrist) / hand_scale
                    shape[:, 1] = -shape[:, 1]
                    shape[:, 2] = -shape[:, 2]

                    # Natural wrist trajectory displacement relative to apex
                    disp = (wrist - wrist_mid) / hand_scale
                    disp_x = float(disp[0]) * 0.45
                    disp_y = float(-disp[1]) * 0.45
                    disp_z = float(-disp[2]) * 0.45

                    frame_coords = (shape - apex_center) + np.array([disp_x, disp_y, disp_z])
                    frame_list.append([[round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)] for p in frame_coords])

                poses[label] = {
                    "type": "greeting",
                    "label": label,
                    "frames": frame_list,
                    "landmarks": frame_list[mid_idx],
                }
            extractor.close()

        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(poses, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Poses canonicas 3D exportadas com sucesso ({len(poses)} itens) para {output_path}")
        return output_path
