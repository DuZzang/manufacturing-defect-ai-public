"""현환 통과 마스크의 우세한 실제 직선 경계만 정리하는 검사 후보.

큰 출력 영역 외곽에서 화면 테두리를 제외하고 강건 직선을 적합한다.
70% 이상이 선의 12.5픽셀 이내에 있으며 화면 크기의 30% 이상 이어질 때만 적용.
그 외에는 기존 마스크 그대로. 질감v2/형상v2/시간정보를 쓰지 않는다.
"""
import cv2
import numpy as np
from hh_polygon_mask_trial import largest_component


def polygon_keep_mask(old_mask):
    main=largest_component(old_mask>0)
    if not main.any():return old_mask.copy(),dict(status='empty',vertices=[])
    h,w=main.shape
    smooth=largest_component(cv2.morphologyEx(main,cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))))
    contours,_=cv2.findContours(smooth,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return old_mask.copy(),dict(status='no_boundary_keep_previous',vertices=[])
    points=max(contours,key=cv2.contourArea)[:,0,:].astype(np.float32)
    points=points[(points[:,0]>2)&(points[:,0]<w-3)&(points[:,1]>2)&(points[:,1]<h-3)]
    if len(points)<100:return old_mask.copy(),dict(status='no_clear_edge_keep_previous',vertices=[])
    vx,vy,x0,y0=cv2.fitLine(points,cv2.DIST_WELSCH,0,.01,.01).flatten()
    distance=np.abs((points[:,0]-x0)*vy-(points[:,1]-y0)*vx)
    inliers=distance<=12.5
    ratio=float(inliers.mean())
    along=(points[inliers,0]-x0)*vx+(points[inliers,1]-y0)*vy
    span=float(np.quantile(along,.95)-np.quantile(along,.05)) if len(along)>1 else 0.
    if ratio<.7 or span<.3*max(h,w):
        return old_mask.copy(),dict(status='edge_not_straight_keep_previous',vertices=[],inlier_ratio=ratio,edge_span=span)
    yy,xx=np.mgrid[:h,:w]
    side=(xx-x0)*vy-(yy-y0)*vx>=0
    candidates=[side,~side]
    def overlap(candidate):return np.count_nonzero(candidate&(main>0))/max(1,np.count_nonzero(candidate|(main>0)))
    chosen=max(candidates,key=overlap)
    result=chosen.astype(np.uint8)*255
    outlines,_=cv2.findContours(result,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    outline=max(outlines,key=cv2.contourArea)
    polygon=cv2.approxPolyDP(outline,1.0,True)
    # 최종 마스크는 직선 반평면 그대로, 표시선만 1픽셀 허용으로 단순화한다.
    return result,dict(status='dominant_straight_edge_trial',vertices=polygon[:,0,:].tolist(),
        inlier_ratio=ratio,edge_span=span,line=[float(v) for v in (vx,vy,x0,y0)])


def self_test():
    old=np.zeros((1000,1000),np.uint8);old[:,300:]=255
    old[100:130,10:30]=255
    old[600:630,280:300]=255
    old[450:460,600:650]=0
    new,info=polygon_keep_mask(old)
    assert info['status']=='dominant_straight_edge_trial'
    assert not new[:,100].any() and (new[:,500]==255).all()
    assert (new[450:460,600:650]==255).all()
    rgb=np.full((1000,1000,3),220,np.uint8);rgb[450:460,600:650]=30
    masked=rgb.copy();masked[new==0]=(124,116,104)
    assert (masked[450:460,600:650]==30).all()
    concave=np.full((1000,1000),255,np.uint8);concave[600:,350:650]=0
    unchanged,info=polygon_keep_mask(concave)
    assert info['status']=='edge_not_straight_keep_previous' and np.array_equal(unchanged,concave)
    assert not polygon_keep_mask(np.zeros_like(old))[0].any()
    assert (polygon_keep_mask(np.full_like(old,255))[0]==255).all()
    print('straight-boundary synthetic checks OK')


if __name__=='__main__':self_test()
