"""팀원 B 베드 마스크의 다각형 후처리 시험. 선별/질감v2/형상v2 변경 없음.

큰 연결 영역 -> 작은 돌기 열림 연산 -> 외곽선 직선 단순화 -> 내부 유지.
볼록 껍질이나 회전 사각형을 강제하지 않아 큰 오목한 가림 영역은 유지할 여지가 있다.
실제 RGB 픽셀을 보정하지 않으며, 마스크만 바꾼다. 검사용이며 미확정 규칙이다.
"""
import cv2
import numpy as np

OPEN_KERNEL = 25
APPROX_PERIMETER_FRACTION = 0.02


def largest_component(binary):
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8),connectivity=8)
    if count <= 1: return np.zeros_like(binary,dtype=np.uint8)
    index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    return (labels==index).astype(np.uint8)


def polygon_keep_mask(old_mask, perimeter_fraction=APPROX_PERIMETER_FRACTION):
    assert old_mask.ndim==2 and set(np.unique(old_mask)) <= {0,255}
    main=largest_component(old_mask>0)
    if not main.any(): return old_mask.copy(),dict(status='empty',vertices=[])
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(OPEN_KERNEL,OPEN_KERNEL))
    smooth=largest_component(cv2.morphologyEx(main,cv2.MORPH_OPEN,kernel))
    if not smooth.any(): return old_mask.copy(),dict(status='opening_removed_candidate_keep_previous',vertices=[])
    contours,_=cv2.findContours(smooth,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    outline=max(contours,key=cv2.contourArea)
    perimeter=cv2.arcLength(outline,True)
    polygon=cv2.approxPolyDP(outline,perimeter_fraction*perimeter,True)
    if len(polygon)<3: return old_mask.copy(),dict(status='invalid_polygon_keep_previous',vertices=[])
    result=np.zeros_like(old_mask)
    cv2.fillPoly(result,[polygon],255)
    return result,dict(status='polygon_trial',vertices=polygon[:,0,:].tolist(),
        opening_kernel=OPEN_KERNEL,epsilon_px=perimeter_fraction*perimeter)


def self_test():
    old=np.zeros((1000,1000),np.uint8)
    square=np.array([[350,0],[999,0],[999,999],[150,999]],np.int32)
    cv2.fillPoly(old,[square],255)
    old[400:430,450:520]=0        # 안쪽 어두운 틈: 마스크는 복구하되 원본 어두운 픽셀은 그대로
    old[120:150,10:30]=255       # 베드 속 별도 섬
    old[700:720,0:20]=255        # 화면 끝에 붙은 별도 섬
    fixed,info=polygon_keep_mask(old)
    assert info['status']=='polygon_trial'
    assert (fixed[400:430,450:520]==255).all()
    assert not fixed[120:150,10:30].any() and not fixed[700:720,0:20].any()
    assert fixed[500,800]==255 and fixed[500,50]==0
    rgb=np.full((1000,1000,3),220,np.uint8);rgb[400:430,450:520]=30
    masked=rgb.copy();masked[fixed==0]=(124,116,104)
    assert (masked[400:430,450:520]==30).all()
    # 큰 오목한 가림: 볼록 도형으로 강제하지 않아야 함.
    concave=np.full((1000,1000),255,np.uint8);concave[600:,350:650]=0
    candidate,_=polygon_keep_mask(concave)
    assert candidate[900,500]==0
    empty,_=polygon_keep_mask(np.zeros_like(old));assert not empty.any()
    full,_=polygon_keep_mask(np.full_like(old,255));assert (full==255).all()
    print('synthetic checks OK: detached sparkle, dark gap, concave occlusion, empty/full')


if __name__=='__main__':self_test()
