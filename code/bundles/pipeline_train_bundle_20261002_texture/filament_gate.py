"""질감 기준 필라멘트 게이트 (2026-10-02, 팀원)

고정 크롭 안에서 "흰 출력물이 보이는 면적 비율"을 계산하고, 기준(기본 0.1) 미만이면 판정 보류.

기존 게이트(팀원 A의 Otsu 밝은 픽셀 비율)는 반짝이 베드를 필라멘트로 잡아서
베드만 찍힌 사진도 0.27~0.51이 나왔음. 이 게이트는 밝기 대신 '질감'을 봄:
  - 흰 출력물: 매끈함 (주변 9x9 밝기 표준편차 2~5)
  - 반짝이 베드: 들쭉날쭉 (표준편차 20~30)

계산 순서
  1. 크롭(1000x1000)을 250x250으로 축소 (INTER_AREA)
  2. 픽셀마다 주변 9x9의 밝기 평균·표준편차, 채도 평균 계산
  3. 표준편차 < 10 (매끈) & 밝기 평균 > 110 (밝음) & 채도 평균 < 80 (색 옅음) -> 출력물 후보
  4. 7x7 원형 열기·닫기로 자잘한 조각 정리, 화면의 2% 미만 덩어리는 버림 (반짝이 뭉침 제거)
  5. 남은 면적 비율 = filament_ratio (0~1). 베드만 있으면 0 근처, 출력물이 화면을 채우면 1 근처

사용법
  from filament_gate import crop_roi, filament_ratio, gate
  crop = crop_roi(img)                    # 원본 1600x1200 -> 고정 크롭 1000x1000
  r = filament_ratio(crop)                # RGB 입력 기본 (PIL, 본인 pipeline.py와 같음)
  r = filament_ratio(crop, bgr=True)      # cv2.imread로 읽은 BGR 입력 (팀원 A 코드)
  ok, r = gate(crop, threshold=0.1)       # ok=False면 판정 보류

필요 패키지: numpy, opencv-python (Colab 기본 설치됨)
"""
import cv2
import numpy as np

ROI = (0, 250, 1000, 1250)       # ymin, xmin, ymax, xmax (팀원 A·본인과 같은 고정 크롭)
THRESHOLD = 0.1                  # 이 비율 미만이면 판정 보류
SIZE = 250                       # 계산용 축소 크기
WIN = 9                          # 질감을 보는 창 크기 (축소 이미지 기준)
SD_T, MEAN_T, SAT_T = 10, 110, 80
MIN_BLOB = 0.02                  # 화면의 2% 미만 덩어리는 버림
_K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def crop_roi(img):
    """원본 프레임 -> 고정 크롭. RGB/BGR 상관없음 (자르기만 함)."""
    y0, x0, y1, x1 = ROI
    assert img.shape[0] >= y1 and img.shape[1] >= x1, f"원본 크기가 크롭보다 작음: {img.shape}"
    return img[y0:y1, x0:x1]


def texture_mask(crop, bgr=False):
    """출력물로 판단한 영역 마스크 (SIZE x SIZE, 0/1)와 축소된 크롭(입력과 같은 색 순서)을 돌려줌."""
    crop = np.ascontiguousarray(crop, dtype=np.uint8)   # torch permute().numpy() 같은 비연속 배열도 받기
    small = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY).astype(np.float32)
    s = cv2.cvtColor(small, cv2.COLOR_BGR2HSV if bgr else cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
    m = cv2.blur(g, (WIN, WIN))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (WIN, WIN)) - m * m, 0))
    raw = ((sd < SD_T) & (m > MEAN_T) & (cv2.blur(s, (WIN, WIN)) < SAT_T)).astype(np.uint8)
    raw = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_OPEN, _K7), cv2.MORPH_CLOSE, _K7)
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw)
    keep = np.zeros_like(raw)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= MIN_BLOB * SIZE * SIZE:
            keep[lab == i] = 1
    return keep, small


def filament_ratio(crop, bgr=False):
    """출력물 면적 비율 (0~1). 본인 pipeline.py의 filament_ratio(rgb_crop)와 같은 형태로 쓸 수 있음."""
    return float(texture_mask(crop, bgr)[0].mean())


def gate(crop, threshold=THRESHOLD, bgr=False):
    """(통과 여부, 비율). 통과 못 하면 판정 보류."""
    r = filament_ratio(crop, bgr)
    return r >= threshold, r


def overlay(crop, bgr=False, color=(255, 40, 40), alpha=0.55):
    """확인용: 출력물로 판단한 영역을 빨간색으로 칠한 RGB 이미지 (SIZE x SIZE)."""
    keep, small = texture_mask(crop, bgr)
    rgb = small[:, :, ::-1].copy() if bgr else small.copy()
    k = keep == 1
    rgb[k] = ((1 - alpha) * rgb[k] + alpha * np.array(color)).astype(np.uint8)
    return rgb


def otsu_ratio(crop, bgr=False):
    """비교용: 기존 게이트 값 (팀원 A의 check_filament_presence와 같은 계산, Otsu 밝은 픽셀 비율)."""
    g = cv2.cvtColor(np.ascontiguousarray(crop, dtype=np.uint8), cv2.COLOR_BGR2GRAY if bgr else cv2.COLOR_RGB2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((b > 0).mean())


def _self_check():
    # 매끈한 흰 면 -> 거의 1, 들쭉날쭉한 반짝이 -> 0
    flat = np.full((1000, 1000, 3), 200, np.uint8)
    rng = np.random.default_rng(0)
    glitter = (rng.random((1000, 1000, 3)) > 0.7).astype(np.uint8) * 230
    assert filament_ratio(flat) > 0.95, filament_ratio(flat)
    assert filament_ratio(glitter) < 0.05, filament_ratio(glitter)


if __name__ == "__main__":
    # python filament_gate.py 사진.jpg [사진2.jpg ...]  -> 사진별 비율과 통과 여부 출력
    import sys
    _self_check()
    for p in sys.argv[1:]:
        img = cv2.imread(p)
        ok, r = gate(crop_roi(img), bgr=True)
        print(f"{p}\tratio={r:.3f}\t{'통과' if ok else '보류'}\t(otsu={otsu_ratio(crop_roi(img), bgr=True):.3f})")
    if len(sys.argv) == 1:
        print("self-check ok")
