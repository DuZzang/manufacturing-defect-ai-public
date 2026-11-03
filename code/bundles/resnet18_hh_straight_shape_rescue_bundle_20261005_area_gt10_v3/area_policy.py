"""단일 면적 정책: 고정 크롭의 출력물 마스크 면적이 10%를 초과해야 사용한다."""
import math

MIN_AREA = 0.1
PIXELS = 1000 * 1000


def eligible(row, minimum=MIN_AREA):
    if not math.isfinite(minimum) or not MIN_AREA <= minimum < 1:
        raise ValueError('면적 하한은 0.1 이상 1 미만이어야 한다. 10% 이하 제외를 해제할 수 없다.')
    if row['cohort'] not in {'primary', 'rescue'}:
        return False
    area = float(row['mask_area'])
    if not math.isfinite(area) or not 0 <= area <= 1:
        raise ValueError(f"마스크 면적 오류: {row['source']}")
    return area > minimum
