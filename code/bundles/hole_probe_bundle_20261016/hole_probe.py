"""질감 마스크 '구멍' 의존 확인: 이미 학습된 ConvNeXt로 추론만 한다 (학습 없음).

구멍 = 질감 마스크가 회색으로 지운 영역 중, 재활용 마스크(통과 사진=베드 지도, 재활용 사진=형상 v2)가 출력물로 보는 곳이고
      재활용 마스크의 바깥 회색과 이어져 있지 않은 덩어리 (= 출력물 안쪽에 갇힌 회색).
입력 변형 (공통 3,039장, 고정 크롭 1000×1000 -> 512, 학습 때 캐시와 같은 방식)
  tex      : 질감 마스크 그대로 (질감 모델의 원래 입력)
  fill     : tex에서 구멍만 원본 화소로 되돌림
  rec      : 재활용 마스크 (대조 모델의 원래 입력)
  add_tex  : 정상 사진의 tex에 언더필 사진의 구멍 모양을 회색으로 옮겨 칠함 (결함 없음)
  add_rec  : 정상 사진의 rec에 같은 구멍 모양을 회색으로 칠함 (대조 모델용)
모델 (각 사진은 OOF였던 fold 모델로만 채점)
  질감×질감  = 10/05 run의 v3_texture,  2단계×질감 = 10/13 run의 v4_s2_texture  -> tex / fill / add_tex
  재활용×재활용 = 10/05 run의 v2 (대조)                                         -> rec / tex / fill / add_rec
재현 확인: 질감 모델의 tex 점수, 대조 모델의 rec 점수가 저장된 OOF 점수와 같아야 한다 (같지 않으면 입력이 학습과 다름).
"""
import argparse, csv, json, math, sys, time
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

GRAY = (124, 116, 104)
FAMILIES = [('질감×질감', 'old', 'v3_texture', ('tex', 'fill', 'add_tex'), 'tex'),
            ('2단계×질감', 's2', 'v4_s2_texture', ('tex', 'fill', 'add_tex'), 'tex'),
            ('재활용×재활용', 'old', 'v2', ('rec', 'tex', 'fill', 'add_rec'), 'rec')]
VARIANTS = ('tex', 'fill', 'rec', 'add_tex', 'add_rec')


def rd(p):
    with open(p, encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def holes_of(T, R):
    """T, R: bool 1000×1000 (True=남김). 질감이 지웠지만 재활용 마스크 바깥 회색과 이어지지 않은 덩어리."""
    notT = (~T).astype(np.uint8)
    n, lab = cv2.connectedComponents(notT, connectivity=8)
    touching = np.unique(lab[(~R) & (~T)])
    return (lab > 0) & ~np.isin(lab, touching)


def self_test():
    T = np.ones((50, 50), bool); T[:, :10] = False; T[20:25, 20:25] = False       # 바깥 회색 + 안쪽 구멍
    R = np.ones((50, 50), bool); R[:, :10] = False
    h = holes_of(T, R)
    assert h[20:25, 20:25].all() and not h[:, :10].any() and h.sum() == 25
    print('self-test OK: 안쪽 구멍만 찾고 바깥 회색은 제외')


def to512(rgb):
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    return TF.resize(t, [512, 512], interpolation=v2.InterpolationMode.BILINEAR, antialias=True).numpy()


def paint(rgb, keep):
    v = rgb.copy(); v[~keep] = GRAY; return v


def main():
    ap = argparse.ArgumentParser()
    for k in ('grid', 'texture', 'bundle', 'originals', 'old_run', 's2_run', 'cache', 'output'):
        ap.add_argument('--' + k.replace('_', '-'), type=Path, required=k not in ('old_run', 's2_run'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--limit', type=int, default=None, help='로컬 점검용: 앞 N장만')
    ap.add_argument('--repro-tol', type=float, default=.02)
    ap.add_argument('--no-repro-check', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, str(a.bundle)); import baseline_classifier as original
    a.output.mkdir(parents=True, exist_ok=True); a.cache.mkdir(parents=True, exist_ok=True)

    sel = {r['source']: r for r in rd(a.grid/'selections.csv')}
    tex = {r['source']: r for r in rd(a.texture/'texture_manifest.csv')}
    fold = {r['source'].replace('\\', '/'): int(r['fold']) for r in rd(a.grid/'folds.csv') if r['role'] == 'val'}
    common = sorted(s for s, r in sel.items() if r['stage2'] == '1' and r['recycle'] == '1' and tex[s]['texture_gate_accepted'] == '1')
    assert len(common) == 3039
    common = common[::max(1, len(common) // a.limit)][:a.limit] if a.limit else common   # 로컬 점검: 두 클래스가 섞이게 등간격
    idx = {s: i for i, s in enumerate(common)}
    label = {s: int(sel[s]['label_id']) for s in common}

    # ---- 1) 마스크·구멍 (1000×1000)
    def masks(s):
        T = np.asarray(Image.open(a.texture/tex[s]['mask_path'])) > 0
        R = np.asarray(Image.open(a.grid/(sel[s]['shape_mask'] or sel[s]['bedmap_mask']))) > 0
        return T, R
    t0 = time.time(); hole_area, donors = {}, {}
    for s in common:
        T, R = masks(s); h = holes_of(T, R); hole_area[s] = float(h.mean())
        if label[s] == 1 and hole_area[s] >= .02:
            donors[s] = np.packbits(h)
    donor_list = sorted(donors, key=lambda s: int(sel[s]['frame_index']))
    normals = sorted([s for s in common if label[s] == 0], key=lambda s: int(sel[s]['frame_index']))
    pair = {s: donor_list[i % len(donor_list)] for i, s in enumerate(normals)} if donor_list else {}
    print(f'구멍 계산 {time.time()-t0:.0f}s | 구멍≥2% 언더필(옮겨 심을 모양) {len(donor_list)}장', flush=True)

    # ---- 2) 입력 캐시 (512, uint8)
    marker = a.cache/'complete.json'
    arr = {v: np.lib.format.open_memmap(a.cache/f'{v}.npy', mode='r+' if marker.exists() else 'w+', dtype=np.uint8,
                                        shape=(len(common), 3, 512, 512)) for v in VARIANTS}
    added = {}
    if not marker.exists():
        for i, s in enumerate(common):
            with Image.open(a.originals/s) as im:
                assert im.size == (1600, 1200)
                rgb = np.asarray(im.convert('RGB').crop((250, 0, 1250, 1000)))
            T, R = masks(s); h = holes_of(T, R)
            arr['tex'][i] = to512(paint(rgb, T)); arr['fill'][i] = to512(paint(rgb, T | h)); arr['rec'][i] = to512(paint(rgb, R))
            if s in pair:
                d = np.unpackbits(donors[pair[s]])[:1000 * 1000].reshape(1000, 1000).astype(bool)
                arr['add_tex'][i] = to512(paint(rgb, T & ~d)); arr['add_rec'][i] = to512(paint(rgb, R & ~d))
                added[s] = dict(donor=pair[s], added_tex=float((T & d).mean()), added_rec=float((R & d).mean()))
            if (i + 1) % 500 == 0: print(f'입력 {i+1}/{len(common)} ({time.time()-t0:.0f}s)', flush=True)
        for x in arr.values(): x.flush()
        json.dump(dict(n=len(common), added=added), open(marker, 'w', encoding='utf-8'))
    else:
        added = json.load(open(marker, encoding='utf-8'))['added']

    # ---- 3) 추론
    assert torch.cuda.is_available(), 'GPU 런타임이 필요합니다'
    dev = torch.device('cuda'); model, _, _ = original.build_model('convnext_tiny'); model.to(dev).eval()
    runs = {'old': a.old_run, 's2': a.s2_run}
    oof = {}
    for fam, run, ver, _, _ in FAMILIES:
        p = runs[run]/'oof_predictions.csv'
        if p.exists():
            for r in rd(p):
                if r['version'] == ver: oof[(ver, int(r['seed']), r['source'].replace('\\', '/'))] = float(r['probability_underfill'])

    @torch.no_grad()
    def predict(variant, ids):
        out = []
        for b in range(0, len(ids), 64):
            x = torch.from_numpy(np.stack([arr[variant][idx[s]] for s in ids[b:b + 64]]))
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(original.to_input(x, dev))
            out += torch.softmax(logits.float(), 1)[:, 1].tolist()
        return out

    rows, missing = [], []
    for fam, run, ver, variants, native in FAMILIES:
        for seed in a.seeds:
            for k in range(5):
                w = runs[run]/ver/f'seed{seed}'/f'fold{k}'/'model_state.pt'
                ids = [s for s in common if fold[s] == k]
                if not w.exists() or not ids:
                    missing.append(str(w)); continue
                model.load_state_dict(torch.load(w, map_location='cpu')); model.eval()
                for v in variants:
                    use = [s for s in ids if s in added] if v.startswith('add') else ids
                    for s, prob in zip(use, predict(v, use)):
                        rows.append(dict(family=fam, version=ver, seed=seed, fold=k, source=s, label_id=label[s], variant=v,
                                         prob=prob, pred=int(prob >= .5), hole_area=round(hole_area[s], 5),
                                         added_area=round(added[s]['added_tex' if v == 'add_tex' else 'added_rec'], 5) if v.startswith('add') else '',
                                         oof_prob=oof.get((ver, seed, s), '') if v == native else ''))
                print(f'{fam} seed{seed} fold{k} 완료 ({time.time()-t0:.0f}s)', flush=True)
    with open(a.output/'probe_predictions.csv', 'w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)

    # ---- 4) 재현 확인 + 요약
    repro = defaultdict(list)
    for r in rows:
        if r['oof_prob'] != '': repro[(r['family'], r['seed'])].append(abs(r['prob'] - r['oof_prob']))
    rep = [dict(family=f, seed=s, n=len(d), max_abs_diff=round(max(d), 5), mean_abs_diff=round(float(np.mean(d)), 6)) for (f, s), d in repro.items()]
    ok = bool(rep) and all(r['max_abs_diff'] <= a.repro_tol for r in rep)
    get = defaultdict(dict)
    for r in rows: get[(r['family'], r['seed'], r['variant'])][r['source']] = r['prob']
    bins = [('<2%', 0, .02), ('2~10%', .02, .10), ('>=10%', .10, 2)]
    A, B = [], []
    for fam, _, _, variants, native in FAMILIES:
        base_u = 'tex'                                       # 구멍이 있는 입력 (대조 모델에게는 낯선 입력)
        for seed in a.seeds:
            pt, pf = get[(fam, seed, base_u)], get[(fam, seed, 'fill')]
            for bn, lo, hi in bins:
                ids = [s for s in pt if label[s] == 1 and lo <= hole_area[s] < hi and s in pf]
                if ids:
                    A.append(dict(family=fam, seed=seed, 구멍구간=bn, 언더필=len(ids),
                                  평균확률_구멍있음=round(float(np.mean([pt[s] for s in ids])), 4), 평균확률_구멍메움=round(float(np.mean([pf[s] for s in ids])), 4),
                                  미탐_구멍있음=sum(pt[s] < .5 for s in ids), 미탐_구멍메움=sum(pf[s] < .5 for s in ids),
                                  메웠더니_정상으로바뀜=sum(pt[s] >= .5 and pf[s] < .5 for s in ids)))
            base_n, add_v = native, ('add_rec' if native == 'rec' else 'add_tex')
            pb, pa = get[(fam, seed, base_n)], get[(fam, seed, add_v)]
            ids = [s for s in pa if s in pb]
            if ids:
                B.append(dict(family=fam, seed=seed, 정상=len(ids), 평균옮긴구멍=round(float(np.mean([added[s]['added_tex' if add_v == 'add_tex' else 'added_rec'] for s in ids])), 4),
                              평균확률_원래=round(float(np.mean([pb[s] for s in ids])), 4), 평균확률_구멍추가=round(float(np.mean([pa[s] for s in ids])), 4),
                              오탐_원래=sum(pb[s] >= .5 for s in ids), 오탐_구멍추가=sum(pa[s] >= .5 for s in ids),
                              추가했더니_언더필로바뀜=sum(pb[s] < .5 and pa[s] >= .5 for s in ids)))
    for name, data in (('A_fill_holes.csv', A), ('B_add_holes.csv', B), ('repro_check.csv', rep)):
        if data:
            with open(a.output/name, 'w', encoding='utf-8-sig', newline='') as f:
                wr = csv.DictWriter(f, list(data[0])); wr.writeheader(); wr.writerows(data)
    info = dict(reproduction_ok=ok, repro_tol=a.repro_tol, n_common=len(common), donors=len(donor_list), missing_weights=missing,
                seconds=round(time.time() - t0), note='재현 실패 시 A/B 결과를 해석하지 말 것')
    json.dump(info, open(a.output/'run_info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False)[:800])
    if not ok and not a.no_repro_check:
        sys.exit('재현 확인 실패: 저장된 OOF 점수와 다릅니다. repro_check.csv를 확인하세요.')


if __name__ == '__main__':
    if '--self-test' in sys.argv: self_test()
    else: main()
