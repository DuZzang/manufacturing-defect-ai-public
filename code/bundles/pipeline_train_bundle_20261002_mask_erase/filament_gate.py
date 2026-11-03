"""질감 기준 필라멘트 게이트 + 베드 마스킹 v2 (2026-10-02, 팀원)

하는 일 (고정 크롭 1000x1000 하나에 대해)
  1. 출력물 영역 찾기 -> 면적 비율(ratio)
  2. 게이트: ratio < 0.1 이면 판정 보류 (베드만 찍힌 사진)
  3. 마스킹: 출력물 바깥(베드)을 회색으로 칠한 크롭을 돌려줌 -> 모델 입력용
     (모델이 '반짝이 = 언더필'로 판단하는 문제를 막기 위함)

출력물 영역 찾는 순서 (250x250으로 줄여서 계산)
  a. 픽셀마다 주변 9x9 밝기 표준편차(질감)와 평균 밝기 계산
  b. 표준편차 < 10 (매끈) & 밝기 > 110 -> 출력물 후보
     (v1의 채도 조건은 삭제: 360장 시험에서 빼도 결과 동일, 자동으로 정하면 흰 출력물끼리 갈라버림)
  c. 7x7 열기·닫기, 화면 2% 미만 덩어리 버림
  d. 흔들린 베드 얼룩 제거: 덩어리의 (표준편차 중앙값 >= 5) & (밝기 중앙값 < 155) & (볼록도 < 0.8)
     -> 셋 다 해당하면 버림. 번진 반짝이는 덜 매끈하고 어둡고 모양이 들쭉날쭉함
  e. 남은 덩어리마다 바깥 윤곽 안쪽을 전부 채움 -> 언더필의 틈(결함 증거)이 회색으로 지워지지 않게
  f. 마스킹 때는 윤곽을 2픽셀(원본 기준 약 8픽셀) 넓혀서 테두리가 잘리지 않게 함

사용법
  from filament_gate import crop_roi, filament_ratio, gate, mask_bed
  crop = crop_roi(img)                     # 원본 1600x1200 -> 고정 크롭 1000x1000
  r = filament_ratio(crop)                 # RGB 기본 (두현님 pipeline.py와 같은 형태)
  ok, r = gate(crop, bgr=True)             # cv2로 읽은 BGR이면 bgr=True. ok=False면 판정 보류
  masked, r = mask_bed(crop, bgr=True)     # 베드를 회색 칠한 크롭 (입력과 같은 크기·색 순서)

필요 패키지: numpy, opencv-python (Colab 기본 설치됨)
"""
import cv2
import numpy as np

VERSION = "v2-mask"
ROI = (0, 250, 1000, 1250)       # ymin, xmin, ymax, xmax
THRESHOLD = 0.1                  # 출력물 면적 이 비율 미만이면 판정 보류
SIZE, WIN = 250, 9               # 계산용 축소 크기, 질감 보는 창
SD_T, MEAN_T = 10, 110           # 출력물 후보: 매끈 & 밝음
MIN_BLOB = 0.02                  # 화면 2% 미만 덩어리 버림
BLUR_SD, BLUR_MEAN, BLUR_SOL = 5, 155, 0.8   # 흔들린 베드 얼룩 판정 (셋 다 해당하면 버림)
MARGIN = 2                       # 마스킹 여유 (축소 기준 픽셀)
GRAY_RGB = (124, 116, 104)       # ImageNet 평균색 -> 정규화하면 거의 0
_K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_KM = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MARGIN + 1, 2 * MARGIN + 1))


def crop_roi(img):
    """원본 프레임 -> 고정 크롭. 색 순서 상관없음."""
    y0, x0, y1, x1 = ROI
    assert img.shape[0] >= y1 and img.shape[1] >= x1, f"원본 크기가 크롭보다 작음: {img.shape}"
    return img[y0:y1, x0:x1]


def object_mask(crop, bgr=False):
    """출력물 영역 마스크 (SIZE x SIZE, 0/1, 윤곽 안 채움)."""
    crop = np.ascontiguousarray(crop, dtype=np.uint8)
    small = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = cv2.blur(g, (WIN, WIN))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (WIN, WIN)) - m * m, 0))
    raw = ((sd < SD_T) & (m > MEAN_T)).astype(np.uint8)
    raw = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_OPEN, _K7), cv2.MORPH_CLOSE, _K7)
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw)
    out = np.zeros_like(raw)
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] < MIN_BLOB * SIZE * SIZE:
            continue
        comp = lab == j
        cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        sol = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        if np.median(sd[comp]) >= BLUR_SD and np.median(m[comp]) < BLUR_MEAN and sol < BLUR_SOL:
            continue                                    # 흔들린 베드 얼룩
        cv2.drawContours(out, cs, -1, 1, thickness=-1)  # 바깥 윤곽 안쪽 전부 채움 (틈 유지)
    return out


def filament_ratio(crop, bgr=False):
    """출력물 면적 비율 (0~1). 두현님 filament_ratio(rgb_crop)와 같은 형태."""
    return float(object_mask(crop, bgr).mean())


def gate(crop, threshold=THRESHOLD, bgr=False):
    """(통과 여부, 비율). 통과 못 하면 판정 보류."""
    r = filament_ratio(crop, bgr)
    return r >= threshold, r


def mask_bed(crop, bgr=False):
    """베드(출력물 바깥)를 회색으로 칠한 크롭과 비율. 크기·색 순서는 입력과 같음."""
    crop = np.ascontiguousarray(crop, dtype=np.uint8)
    om = object_mask(crop, bgr)
    keep = cv2.dilate(om, _KM)
    keep = cv2.resize(keep, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = crop.copy()
    out[~keep] = GRAY_RGB[::-1] if bgr else GRAY_RGB
    return out, float(om.mean())


def overlay(crop, bgr=False, color=(255, 40, 40), alpha=0.55):
    """확인용: 출력물로 판단한 영역을 빨간색으로 칠한 RGB 이미지 (SIZE x SIZE)."""
    om = object_mask(crop, bgr)
    small = cv2.resize(np.ascontiguousarray(crop, dtype=np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rgb = small[:, :, ::-1].copy() if bgr else small.copy()
    k = om == 1
    rgb[k] = ((1 - alpha) * rgb[k] + alpha * np.array(color)).astype(np.uint8)
    return rgb


def otsu_ratio(crop, bgr=False):
    """비교용: 기존 게이트 값 (원욱님 check_filament_presence와 같은 Otsu 밝은 픽셀 비율)."""
    g = cv2.cvtColor(np.ascontiguousarray(crop, dtype=np.uint8), cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((b > 0).mean())


def _self_check():
    flat = np.full((1000, 1000, 3), 200, np.uint8)
    rng = np.random.default_rng(0)
    glitter = (rng.random((1000, 1000, 3)) > 0.7).astype(np.uint8) * 230
    assert filament_ratio(flat) > 0.95, filament_ratio(flat)
    assert filament_ratio(glitter) < 0.05, filament_ratio(glitter)
    half = glitter.copy(); half[500:] = 200
    mk, r = mask_bed(half)
    assert 0.4 < r < 0.6 and (mk[:400] == GRAY_RGB).all() and (mk[600:] == 200).all()


if __name__ == "__main__":
    # python filament_gate.py 사진.jpg ... -> 사진별 비율, 통과/보류, 기존 Otsu 값
    import sys
    _self_check()
    for p in sys.argv[1:]:
        img = cv2.imread(p)
        ok, r = gate(crop_roi(img), bgr=True)
        print(f"{p}\tratio={r:.3f}\t{'통과' if ok else '보류'}\t(otsu={otsu_ratio(crop_roi(img), bgr=True):.3f})")
    if len(sys.argv) == 1:
        print("self-check ok", VERSION)
