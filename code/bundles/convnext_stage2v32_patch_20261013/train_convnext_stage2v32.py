"""ConvNeXt-Tiny: 2단계 선별(v3.2) × {질감 마스크, 베드 지도 마스크}. Colab GPU 학습, 시드별 중간 결과 보존.

2026-10-05 세 전처리 실험(train_convnext_three.py)과 같은 엔진·설정을 쓰고, 바꾼 것은 두 가지다.
  1) 학습 사진: 각 fold 학습 구간 중 2단계 v3.2 통과 사진 (팀원 reference.json 그대로)
  2) 입력 마스크: v4 = 질감 마스크(기존 patch의 texture/masks), v5 = 베드 지도(DINOv2 격자 묶음 masks/bedmap)
업데이트 횟수는 옛 실험 v1(옛 2단계 통과 2,652장 중 면적>10%) 학습 장수 기준 그대로라 옛 결과와 학습 예산이 같다.
평가 사진은 각 fold val 구간 4,000장 전부(선별 밖 사진도 점수를 낸다). 순위 비교는 공통 사진 3,039장에서만 한다.
면적 10% 제외는 하지 않는다(DINOv2 격자와 같은 규칙). 마스크 면적은 기록만 한다.
"""
import argparse
import gc
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

import engine_convnext as engine

VERSIONS = ('v4_s2_texture', 'v5_s2_bedmap')
MASK = {'v4_s2_texture': 'texture', 'v5_s2_bedmap': 'bedmap'}
GRAY = (124, 116, 104)
sha, read_csv, write_csv, save_json = engine.sha, engine.read_csv, engine.write_csv, engine.save_json
# 평가 묶음. common = 2단계 ∩ 재활용 ∩ 질감 게이트 통과 (DINOv2 격자 A와 같은 3,039장)
GROUPS = {
    'common': lambda r: r['common'],
    'common_clean': lambda r: r['common'] and r['clean_old'],
    'common_bed_visible': lambda r: r['common'] and not r['clean_old'],
    'own_stage2': lambda r: r['stage2'],
    'some_pass': lambda r: not r['common'] and (r['stage2'] or r['recycle'] or r['texture_gate']),
    'none_pass': lambda r: not (r['stage2'] or r['recycle'] or r['texture_gate']),
    'all4000': lambda r: True,
}


def load_rows(args):
    man = {r['source'].replace('\\', '/'): r for r in read_csv(args.bundle/'manifest.csv')}
    tex = {r['source']: r for r in read_csv(args.texture/'texture_manifest.csv')}
    sel = read_csv(args.grid/'selections.csv')
    assert len(man) == len(tex) == len(sel) == 4000
    rows = []
    for s in sel:
        m, t = man[s['source']], tex[s['source']]
        assert m['source_sha256'] == t['source_sha256'] and int(m['frame_index']) == int(s['frame_index'])
        r = dict(source=s['source'], label=s['label'], label_id=int(s['label_id']), frame_index=int(s['frame_index']),
                 source_sha256=m['source_sha256'], stage2=int(s['stage2']), recycle=int(s['recycle']),
                 clean_old=int(s['clean_old']), texture_gate=int(t['texture_gate_accepted']),
                 # 업데이트 예산 기준: 옛 실험 v1 학습 사진 = 옛 2단계 통과(primary) 중 마스크 면적 > 10%
                 reference=int(m['cohort'] == 'primary' and float(m['mask_area'] or 0) > .1),
                 texture_mask_path=t['mask_path'], texture_mask_sha256=t['mask_sha256'],
                 texture_mask_area=float(t['texture_mask_area']),
                 bedmap_mask_path=s['bedmap_mask'], bedmap_mask_area=float(s['bedmap_area']))
        r['common'] = int(r['stage2'] and r['recycle'] and r['texture_gate'])
        r['cohort'] = 'stage2' if r['stage2'] else 'held'
        rows.append(r)
    assert sum(r['stage2'] for r in rows) == 3062 and sum(r['common'] for r in rows) == 3039
    assert sum(r['reference'] for r in rows) == len({r['source'] for r in rows if r['reference']})
    return rows


def decides(r, version):
    return bool(r['stage2'])


def split(rows, fold_rows, fold, version):
    roles = {r['source'].replace('\\', '/'): r['role'] for r in fold_rows if int(r['fold']) == fold}
    assert len(roles) == 4000 and set(roles.values()) <= {'train', 'val', 'buffer'}
    if version == 'v1':   # 엔진이 업데이트 횟수를 정할 때만 쓰는 기준 목록
        return [r for r in rows if roles[r['source']] == 'train' and r['reference']], []
    tr, va = [], []
    for r in rows:
        item = dict(r, mask_area=r[MASK[version]+'_mask_area'], mask_status=MASK[version])
        if roles[r['source']] == 'train' and r['stage2']:
            tr.append(item)
        elif roles[r['source']] == 'val':
            va.append(item)
    assert not ({r['source'] for r in tr} & {r['source'] for r in va})
    for cls in ('normal', 'underfill_50'):
        a = np.asarray([r['frame_index'] for r in tr if r['label'] == cls])
        b = np.asarray([r['frame_index'] for r in va if r['label'] == cls])
        assert a.size and b.size and np.min(np.abs(a[:, None]-b[None, :])) > 30
    return tr, va


def prepare_cache(args, rows):
    args.cache.mkdir(parents=True, exist_ok=True)
    config = dict(selections_sha=sha(args.grid/'selections.csv'),
        texture_manifest_sha=sha(args.texture/'texture_manifest.csv'), size=args.size,
        crop=[250, 0, 1250, 1000], resize='tensor bilinear antialias=True', gray=GRAY,
        source_order=[r['source'] for r in rows], adapter_sha=sha(__file__))
    marker = args.cache/'complete.json'
    shape = (len(rows), 3, args.size, args.size)
    if marker.exists():
        assert json.loads(marker.read_text()) == config, '새 설정은 새 캐시 경로를 사용하세요.'
        for name in MASK.values():
            a = np.load(args.cache/f'{name}.npy', mmap_mode='r')
            assert a.shape == shape and a.dtype == np.uint8
        return
    arrays = {name: np.lib.format.open_memmap(args.cache/f'{name}.npy', mode='w+', dtype=np.uint8, shape=shape)
              for name in MASK.values()}
    for i, r in enumerate(rows):
        source = args.originals/r['source']
        assert sha(source) == r['source_sha256']
        with Image.open(source) as im:
            assert im.size == (1600, 1200)
            rgb = np.asarray(im.convert('RGB').crop((250, 0, 1250, 1000)))
        for name, path, ndigits in (('texture', args.texture/r['texture_mask_path'], 12),
                                    ('bedmap', args.grid/r['bedmap_mask_path'], 4)):
            if name == 'texture':
                assert sha(path) == r['texture_mask_sha256']
            with Image.open(path) as im:
                mask = np.asarray(im)
            assert mask.shape == (1000, 1000) and set(np.unique(mask)) <= {0, 255}
            # 베드 지도는 파일 해시가 목록에 없어서 면적(소수 4자리)으로 파일이 맞는지 확인한다
            assert abs(round(float((mask > 0).mean()), ndigits)-r[f'{name}_mask_area']) < 1e-9, (name, r['source'])
            value = rgb.copy()
            value[mask == 0] = GRAY
            arrays[name][i] = TF.resize(torch.from_numpy(value).permute(2, 0, 1), [args.size, args.size],
                interpolation=v2.InterpolationMode.BILINEAR, antialias=True).numpy()
        if (i+1) % 500 == 0:
            print(f'입력 캐시 {i+1}/{len(rows)}', flush=True)
    for a in arrays.values():
        a.flush()
    save_json(marker, config)


class Inputs:
    def __init__(self, cache, rows, version):
        self.data = np.load(cache/f'{MASK[version]}.npy', mmap_mode='r')
        self.index = {r['source']: i for i, r in enumerate(rows)}

    def __getitem__(self, source):
        return torch.from_numpy(self.data[self.index[source]].copy())


engine.split, engine.decides, engine.Inputs = split, decides, Inputs


def aggregate(output, rows):
    info = {r['source']: r for r in rows}
    oof, completed = [], set()
    for p in sorted(output.glob('v*/seed*/fold*/complete.json')):
        cfg = json.loads(p.read_text())['config']
        assert cfg['version'] in VERSIONS
        completed.add((cfg['version'], cfg['seed'], cfg['fold']))
        oof.extend(read_csv(p.parent/'predictions.csv'))
    for r in oof:
        for k in ('seed', 'fold', 'label_id', 'pred', 'pipeline_decides', 'frame_index'):
            r[k] = int(r[k])
        r['probability_underfill'] = float(r['probability_underfill'])
        assert math.isfinite(r['probability_underfill']) and r['pred'] == int(r['probability_underfill'] >= .5)
        g = info[r['source']]
        r.update(stage2=g['stage2'], recycle=g['recycle'], texture_gate=g['texture_gate'],
                 common=g['common'], clean_old=g['clean_old'])
    assert len(oof) == len({(r['version'], r['seed'], r['source']) for r in oof})
    write_csv(output/'oof_predictions.csv', oof)
    summary, pairs, finished = [], [], []
    for seed in sorted({r['seed'] for r in oof}):
        complete = {v: all((v, seed, f) in completed for f in range(5)) for v in VERSIONS}
        maps = {v: {r['source']: r for r in oof if r['version'] == v and r['seed'] == seed} for v in VERSIONS}
        for v in VERSIONS:
            if not complete[v]:
                continue   # 미완료 fold가 섞인 숫자는 내지 않는다
            assert len(maps[v]) == 4000
            for name, rule in GROUPS.items():
                group = [r for r in maps[v].values() if rule(r)]
                summary.append(dict(version=v, seed=seed, group=name, **engine.original.binary_metrics(
                    [r['label_id'] for r in group], [r['probability_underfill'] for r in group])))
        if all(complete.values()):
            finished.append(seed)
            a, b = VERSIONS
            for name in ('common', 'common_clean', 'common_bed_visible'):
                ids = [k for k in maps[a] if GROUPS[name](maps[a][k])]
                ca = [maps[a][k]['pred'] == maps[a][k]['label_id'] for k in ids]
                cb = [maps[b][k]['pred'] == maps[b][k]['label_id'] for k in ids]
                pairs.append(dict(seed=seed, a=a, b=b, group=name, n=len(ids),
                    b_only_correct=sum(y and not x for x, y in zip(ca, cb)),
                    a_only_correct=sum(x and not y for x, y in zip(ca, cb))))
    write_csv(output/'summary.csv', summary)
    write_csv(output/'paired_comparison.csv', pairs)
    means = []
    for v, name in sorted({(r['version'], r['group']) for r in summary}):
        group = [r for r in summary if r['version'] == v and r['group'] == name]
        for metric in ('accuracy', 'underfill_recall', 'normal_recall', 'auroc', 'fp', 'fn'):
            values = [r[metric] for r in group if r[metric] is not None]
            if values:
                means.append(dict(version=v, group=name, metric=metric, seeds=','.join(str(r['seed']) for r in group),
                    n=group[0]['n'], mean=statistics.mean(values),
                    seed_sd=statistics.stdev(values) if len(values) > 1 else None))
    write_csv(output/'across_seeds.csv', means)
    save_json(output/'progress.json', dict(fully_completed_seeds=finished, completed_runs=len(completed),
        planned_runs=30, mean_sd_note='시드 간 표준편차이며 일반화 신뢰구간이 아님'))


def audit(args, rows, folds):
    records, coverage = [], []
    for version in VERSIONS:
        for fold in range(5):
            tr, va = split(rows, folds, fold, version)
            ref, _ = split(rows, folds, fold, 'v1')
            records.append(dict(version=version, fold=fold, train_n=len(tr), val_n=len(va), reference_n=len(ref),
                train_normal=sum(r['label_id'] == 0 for r in tr), train_underfill=sum(r['label_id'] == 1 for r in tr),
                train_min_mask_area=min(r['mask_area'] for r in tr), train_mask_area_le10=sum(r['mask_area'] <= .1 for r in tr),
                per_round_steps=math.ceil(len(ref)/args.batch), total_update_steps=math.ceil(len(ref)/args.batch)*args.rounds))
    for cls in ('normal', 'underfill_50'):
        seq = sorted((r for r in rows if r['label'] == cls), key=lambda r: r['frame_index'])
        best = current = 0
        for r in seq:
            current = 0 if r['stage2'] else current+1
            best = max(best, current)
        coverage.append(dict(selection='stage2_v32', label=cls, total=len(seq), accepted=sum(r['stage2'] for r in seq),
            coverage=sum(r['stage2'] for r in seq)/len(seq), longest_hold_run=best))
    write_csv(args.output/'eligibility_audit.csv', records)
    write_csv(args.output/'coverage.csv', coverage)
    write_csv(args.output/'input_manifest.csv', [{k: r[k] for k in (
        'source', 'label', 'frame_index', 'stage2', 'recycle', 'texture_gate', 'common', 'clean_old', 'reference',
        'texture_mask_area', 'bedmap_mask_area')} for r in rows])
    print(json.dumps(records, ensure_ascii=False), flush=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args)
    folds = read_csv(args.bundle/'folds.csv')
    assert sha(args.bundle/'folds.csv') == sha(args.grid/'folds.csv')
    audit(args, rows, folds)
    protocol = dict(versions=list(VERSIONS), planned_seeds=[0, 1, 2], folds=[0, 1, 2, 3, 4],
        size=args.size, batch=args.batch, rounds=args.rounds,
        selections_sha=sha(args.grid/'selections.csv'), folds_sha=sha(args.bundle/'folds.csv'),
        texture_manifest_sha=sha(args.texture/'texture_manifest.csv'), adapter_sha=sha(__file__),
        engine_sha=sha(engine.__file__), train_selection='stage2 v3.2 (reference.json)',
        val='all photos in fold val block (4,000 over 5 folds)', area_exclusion='none (recorded only)',
        equal_budget='same fold update count as 2026-10-05 run (old v1 train photos)',
        final_selection='fixed last round', threshold=.5,
        limitations=['one print per class', 'not an independent test', 'stage2 judges top 85% of frame, model input is fixed crop',
                     'mask area shortcut unresolved'])
    pp = args.output/'run_settings.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol, '설정 변경: 새 RUN_NAME이 필요합니다.'
    else:
        save_json(pp, protocol)
    if args.verify_only:
        print('10가지 분할·30프레임 buffer·학습 예산 점검 완료. 학습 없음.', flush=True)
        return
    prepare_cache(args, rows)
    if args.prepare_only:
        print('캐시 준비 완료. 학습 없음.', flush=True)
        return
    assert torch.cuda.is_available(), 'Colab GPU를 연결하세요.'
    torch.backends.cudnn.benchmark = True
    save_json(args.output/'environment.json', dict(torch=str(torch.__version__),
        torchvision=__import__('torchvision').__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), fast_nondeterministic=True))
    for seed in args.seeds:
        for fold in args.folds:
            for version in VERSIONS:
                engine.run(args, rows, folds, version, seed, fold)
                aggregate(args.output, rows)
                folder = args.output/version/f'seed{seed}'/f'fold{fold}'
                if (folder/'complete.json').exists() and (folder/'model_state.pt').exists():
                    (folder/'latest.pt').unlink(missing_ok=True)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'texture', 'grid', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--originals', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    args.rescue_min_area = .1   # 엔진 config 기록용(옛 실험과 같은 값). 이 실험은 면적 제외를 하지 않는다.
    if not set(args.seeds) <= {0, 1, 2} or not set(args.folds) <= set(range(5)):
        p.error('시드는 0/1/2, fold는 0..4를 사용하세요.')
    if not args.verify_only and (args.originals is None or args.cache is None):
        p.error('--originals와 --cache가 필요합니다.')
    main(args)
