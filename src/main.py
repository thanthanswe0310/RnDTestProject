# =====================================================
# + NASM 오버헤드 스쿼트 평가
#   주어진 데이터를 사용하여 NASM 오버헤드 스쿼트 평가에 따라 다음 항목을 평가합니다:
#   +++++ 정면도
#   +++++ 무릎이 안쪽 또는 바깥쪽으로 움직입니다
#   +++++ 무릎 편차 정도 출력
#   +++++ 측면도
#   +++++ 로우 백 아치
#   +++++ 요추 아치의 정도 출력
#   +++++ 앞으로 기울어진 몸통
#   +++++ 앞으로 상체를 기울이는 정도를 출력합니다
#
# 시스템 모델:
# 결정론적 신호 처리 전처리를 기반으로 한
# 규칙 기반 생체역학 평가 모델
#
# 파이프라인 개요:
# 1) Crenna 정렬 신호 필터링 (영위상 Butterworth 필터)
# 2) 기하학적 관절 각도 모델링 (벡터 기반 운동학)
# 3) 동작 단계별 분석 (스쿼트 최하단 구간 검출)
# 4) 임계값 기반 NASM 분류 (규칙 기반 판단)
# 5) 3D 운동학 시각화 (해석 가능성 및 검증 목적)
#
# 참고 문헌:
# Crenna, F., Rossi, G. B., & Berardengo, M.
# "Filtering Biomechanical Signals in Movement Analysis"
# =====================================================

# ===============================
# 1. Imports
# ===============================
# 수치 연산, 데이터 처리, 시각화 및
# 결정론적 디지털 신호 처리를 위한 라이브러리

import pandas as pd
import numpy as np
import os

# 현재 시스템은 CPU만 사용 중이며 OpenGL, GPU, 그래픽 드라이버가 없습니다.
# 따라서 시각화를 위해 Matplotlib 라이브러리를 사용합니다.
import matplotlib.pyplot as plt 
import imageio.v2 as imageio
from io import BytesIO
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 2. Paths
# ===============================
# 입력: 3D 관절 좌표가 저장된 Excel 파일
# 출력: 3D 스켈레톤 GIF 시각화 결과
DATA_DIR   = r"D:\RnDTestProject\data" 
OUTPUT_DIR = r"D:\RnDTestProject\test탄탄스웨_output\탄탄스웨_datavisualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 3. Data Ingestion
# ===============================
# Excel 파일로 저장된 모든 모션 캡처 실험 데이터를
#  수집
def get_excel_files(folder):

    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".xlsx")
    ]

# ===============================
# 4. Signal Processing Stage
# ===============================
# Crenna 기준에 따른 결정론적 전처리:
# 영위상 Butterworth 저역통과 필터링
#
# 목적:
# - 고주파 측정 노이즈 제거
# - 시간 정렬 유지 (위상 왜곡 없음)
# - 운동학 신호의 생체역학적 타당성 보존
def crenna_lowpass_filter(df, fs=30.0, cutoff=6.0, order=4):

    """
    Crenna et al.의 생체역학 신호 처리
    권고안을 따르는 영위상 Butterworth
    저역통과 필터
    """

    filtered = df.copy()
    nyq = fs / 2.0
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    for col in df.columns:
        if col == "timestamp":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            signal = df[col].values
            if np.isnan(signal).any():
                continue
              # 전·후방 필터링을 통한 영위상 응답 구현
            if len(signal) > order * 3:
                filtered[col] = filtfilt(b, a, signal)

    return filtered

# ===============================
# 5. Mathematical Utilities
# ===============================
# 파이프라인 전반에서 사용되는 핵심 기하학 연산:
# 관절 운동학을 위한 벡터 기반 각도 계산

def angle_between(v1, v2):

    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    v1, v2 = v1 / n1, v2 / n2
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

# ===============================
# 6. Phase-Specific Movement Analysis
# ===============================
# 스쿼트 동작의 최하단 구간을 식별
#
# 근거:
# NASM 평가는 보상 패턴이 가장 명확히
# 나타나는 최대 하강 지점에서 가장 의미 있음

def bottom_frames(df, y_col="waist_y", pct=0.2):
    min_y = df[y_col].min()
    max_y = df[y_col].max()
    threshold = min_y + pct * (max_y - min_y)
    return df[df[y_col] <= threshold]

# ===============================
# 7. Geometric Joint-Angle Modeling
# ===============================
# 해부학적 관절 벡터에 기반한
# 결정론적 운동학 모델 (머신러닝 미사용)

def knee_deviation(df, side="l"):

     # 전면면에서의 무릎 내·외반(Valgus/Varus) 추정
    hip   = df[[f"{side}_hip_x",   f"{side}_hip_y"]].values
    knee  = df[[f"{side}_knee_x",  f"{side}_knee_y"]].values
    ankle = df[[f"{side}_ankle_x", f"{side}_ankle_y"]].values
    raw = np.array([angle_between(h - k, a - k) for h, k, a in zip(hip, knee, ankle)])
    return 180 - raw

def lumbar_extension(df):

    # 시상면에서의 요추 과신전(허리 아치) 추정
    torso = df[["torso_y", "torso_z"]].values
    waist = df[["waist_y", "waist_z"]].values
    hip_mid_y = (df["l_hip_y"] + df["r_hip_y"]) / 2
    hip_mid_z = (df["l_hip_z"] + df["r_hip_z"]) / 2
    hip_mid = np.column_stack((hip_mid_y, hip_mid_z))
    raw = np.array([angle_between(t - w, h - w) for t, w, h in zip(torso, waist, hip_mid)])
    return 180 - raw

def torso_lean(df):

    # 수직축 대비 몸통 전방 기울기 계산
    torso = df[["torso_y", "torso_z"]].values
    waist = df[["waist_y", "waist_z"]].values
    vertical = np.array([1, 0])
    return np.array([
        angle_between(t - w, vertical) 
        for t, w in zip(torso, waist)
        ]
    )

# ===============================
# 8. Rule-Based NASM Classification
# ===============================
# NASM 기준에 따른 임계값 기반 전문가 규칙
# 머신러닝이 아닌 결정론적 판단 모델

def classify(value, mild, excessive):

    if value <= mild:
        return "NORMAL"
    elif value <= excessive:
        return "MILD"
    else:
        return "FAULT"

# ===============================
# 9. Kinematic Skeleton Definition
# ===============================
# 시각화를 위한 해부학적 관절 연결 구조 정의

SKELETON = [
    ("head", "torso"), ("torso", "waist"),
    ("torso", "l_shoulder"), ("torso", "r_shoulder"),
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("waist", "l_hip"), ("l_hip", "l_knee"), 
    ("l_knee", "l_ankle"),
    ("waist", "r_hip"), 
    ("r_hip", "r_knee"), 
    ("r_knee", "r_ankle"),
]

# ===============================
# 10. 3D Visualization Setup
# ===============================
# 시각화는 해석 가능성과
# 정성적 생체역학 검증을 목적으로 수행

def setup_3d_axes(ax):

    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 3000)
    ax.set_zlim(-1250, 750)
    
    ax.set_xlabel("X (Left–Right)")
    ax.set_ylabel("Z (Front–Back)")
    ax.set_zlabel("Y (Vertical)")

    ax.set_box_aspect((1, 2.75, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=15, azim=-80) 

    # ---  gray background panes ---

    gray = (0.90, 0.90, 0.90, 1.0)
    ax.xaxis.set_pane_color(gray)
    ax.yaxis.set_pane_color(gray)
    ax.zaxis.set_pane_color(gray)
    ax.grid(True, alpha=0.25)
    ax.margins(0)

# ===============================
# 11. 3D 스켈레톤 GIF 생성기
# ===============================
# 프레임별 운동학적 시각화
# 결과 해석 및 보고 지원

def save_skeleton_gif(df, filename, title):

    path = os.path.join(OUTPUT_DIR, filename)
    with imageio.get_writer(path, mode="I", duration=0.08, loop=0) as writer:
        for i in range(len(df)):
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection="3d")

             # 골격 세그먼트 그리기
            for a, b in SKELETON:
                ax.plot(
                    [df.loc[i, f"{a}_x"], df.loc[i, f"{b}_x"]],
                    [df.loc[i, f"{a}_z"], df.loc[i, f"{b}_z"]],
                    [df.loc[i, f"{a}_y"], df.loc[i, f"{b}_y"]],
                    color="blue", linewidth=2.5
                )

            # 관절 마커 그리기
            ax.scatter(
                df.filter(like="_x").iloc[i],
                df.filter(like="_z").iloc[i],
                df.filter(like="_y").iloc[i],
                color="red",s=105, edgecolors="black", linewidths=1.5
            )

            setup_3d_axes(ax)
            fig.suptitle(title, fontsize=14)
            ax.text2D( 0.02, 0.95, 
                      f"Frame {i+1}/{len(df)}", 
                      transform=ax.transAxes, 
                      fontsize=14, 
                      bbox=dict( facecolor="#fff9c4", 
                                alpha=0.75,
                                edgecolor="#333333", 
                                boxstyle="round,pad=0.35" 
                                ) 
                            )   

            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=200)
            plt.close(fig)
            buf.seek(0)
            writer.append_data(imageio.imread(buf))
            buf.close()

    print(f"[확인] 저장된 GIF : {path}")

# ===============================
# 12. 주요 파이프라인 실행
# ===============================
# 종단 간 결정론적 생체역학 분석

excel_files = get_excel_files(DATA_DIR)
print(f"Found {len(excel_files)} files")

for path in excel_files:

    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\nProcessing: {name}")

    # 모션 캡처 데이터 로드
    df = pd.read_excel(path)
    df.columns = df.columns.str.lower().str.strip()

    # (1) 크레나 정렬 신호 전처리
    df = crenna_lowpass_filter(df, fs=30.0, cutoff=6.0)

    # (2) 단계별 분석
    df_bottom = bottom_frames(df)

    # (3) 기하학적 관절 각도 계산
    lk = np.nanmean(knee_deviation(df_bottom, "l"))
    rk = np.nanmean(knee_deviation(df_bottom, "r"))
    lumbar = np.nanmean(lumbar_extension(df_bottom))
    torso  = np.nanmean(torso_lean(df_bottom))

    # (4) 규칙 기반 NASM 분류
    print(f"Left Knee  : {lk:.2f}= {classify(lk, 5, 10)}")
    print(f"Right Knee : {rk:.2f}= {classify(rk, 5, 10)}")
    print(f"Lumbar Ext : {lumbar:.2f}= {classify(lumbar, 10, 20)}")
    print(f"Torso Lean : {torso:.2f}= {classify(torso, 10, 20)}")

    # (5) 3D 운동학적 시각화
    save_skeleton_gif(df, f"{name}_skeleton.gif", f"{name} – Example Data")

print("\nALL FILES PROCESSED SUCCESSFULLY")
