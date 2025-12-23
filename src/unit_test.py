# Unit Test Code | 유닛 테스트 코드**  
#    **KR**
#    - Python 기반 유닛 테스트를 통해 다음 항목 검증:
#      - 신호 필터링
#      - 기하학적 관절 각도 계산
#      - NASM 규칙 기반 분류
#      - 3D 스켈레톤 시각화 실행 여부

import unittest
import numpy as np
import pandas as pd
import os
import tempfile

from main import (
    crenna_lowpass_filter,
    angle_between,
    bottom_frames,
    knee_deviation,
    lumbar_extension,
    torso_lean,
    classify,
    save_skeleton_gif
)

class TestNASMFunctions(unittest.TestCase):

    def setUp(self):
        n = 5

        # Skeleton에 필요한 모든 joint 목록
        joints = [
            "head", "torso", "waist",
            "l_shoulder", "r_shoulder",
            "l_elbow", "r_elbow",
            "l_wrist", "r_wrist",
            "l_hip", "r_hip",
            "l_knee", "r_knee",
            "l_ankle", "r_ankle"
        ]

        data = {
            "timestamp": np.arange(n)
        }

        # 모든 joint에 대해 x,y,z 생성
        for j in joints:
            data[f"{j}_x"] = np.zeros(n)
            data[f"{j}_y"] = np.linspace(0, 2, n)
            data[f"{j}_z"] = np.zeros(n)

        self.df = pd.DataFrame(data)

    # ----------------------------
    # 신호 필터링
    # ----------------------------
    def test_crenna_lowpass_filter(self):
        filtered = crenna_lowpass_filter(self.df, fs=30.0, cutoff=6.0)
        self.assertEqual(len(filtered), len(self.df))
        np.testing.assert_array_equal(
            filtered["timestamp"].values,
            self.df["timestamp"].values
        )

    # ----------------------------
    # 각도 유틸리티
    # ----------------------------
    def test_angle_between(self):
        self.assertAlmostEqual(angle_between([1, 0], [0, 1]), 90.0)
        self.assertAlmostEqual(angle_between([1, 0], [1, 0]), 0.0)
        self.assertAlmostEqual(angle_between([1, 0], [-1, 0]), 180.0)
        self.assertTrue(np.isnan(angle_between([0, 0], [1, 1])))

    # ----------------------------
    # 하단 프레임 감지
    # ----------------------------
    def test_bottom_frames(self):
        bottom = bottom_frames(self.df, y_col="waist_y", pct=0.2)
        self.assertTrue(len(bottom) > 0)

    # ----------------------------
    # 생체역학 지표
    # ----------------------------
    def test_knee_deviation(self):
        angles = knee_deviation(self.df, side="l")
        self.assertIsInstance(angles, np.ndarray)
        self.assertEqual(len(angles), len(self.df))

    def test_lumbar_extension(self):
        angles = lumbar_extension(self.df)
        self.assertIsInstance(angles, np.ndarray)
        self.assertEqual(len(angles), len(self.df))

    def test_torso_lean(self):
        angles = torso_lean(self.df)
        self.assertIsInstance(angles, np.ndarray)
        self.assertEqual(len(angles), len(self.df))

    # ----------------------------
    # NASM Classification
    # ----------------------------
    def test_classify(self):
        self.assertEqual(classify(3, 5, 10), "NORMAL")
        self.assertEqual(classify(7, 5, 10), "MILD")
        self.assertEqual(classify(15, 5, 10), "FAULT")

    # ----------------------------
    # GIF 생성
    # ----------------------------
    def test_save_skeleton_gif(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from main import OUTPUT_DIR
            original_dir = OUTPUT_DIR

            try:
                # 임시 디렉토리로 출력 경로 변경
                import main
                main.OUTPUT_DIR = tmpdir

                save_skeleton_gif(self.df, "test.gif", "Unit Test Skeleton")
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "test.gif")))
            finally:
                main.OUTPUT_DIR = original_dir


if __name__ == "__main__":
    unittest.main()
