"""세 전처리의 ConvNeXt-Tiny 비교. Colab GPU 학습, 시드별 중간 결과 보존."""
import argparse
import gc
import itertools
import json
import math
import shutil
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

import engine_convnext as engine

VERSIONS = ('v1', 'v2', 'v3_texture')
sha, read_csv, write_csv, save_json = engine.sha, engine.read_csv, engine.write_csv, engine.save_json
MINIMUM = .1


def load_rows(args):
    texture = {r['source']: r for r in read_csv(args.texture/'texture_manifest.csv')}
    full = read_csv(args.bundle/'manifest.csv')
    assert len(full) == len(texture) == 4000
    rows = []
    for raw in full:
        r = dict(raw)
        r['source'] = r['source'].replace('\\', '/')
        t = texture[r['source']]
        assert r['source_sha256'] == t['source_sha256']
        hh = engine.eligible(r, args.rescue_min_area)
        ti = bool(int(t['texture_input_eligible'])) and float(t['texture_mask_area']) > args.rescue_min_area
        tt = bool(int(t['texture_train_eligible'])) and ti
        if not (hh or tt):
            continue
        assert not int(t['common_excluded'])
        r.update(label_id=int(r['label_id']), frame_index=int(r['frame_index']),
            hh_eligible=hh, texture_input=ti, texture_train=tt,
            hh_mask_area=float(r['mask_area'] or 0), mask_area=float(r['mask_area'] or 0),
            texture_mask_area=float(t['texture_mask_area']), texture_gate_ratio=float(t['texture_gate_ratio']),
            texture_mask_path=t['mask_path'], texture_mask_sha256=t['mask_sha256'])
        if not hh:
            r['cohort'] = 'texture_only'
        rows.append(r)
    assert len(rows) == len({r['source'] for r in rows})
    return rows, full


def decides(r, version):
    if version == 'v3_texture':
        return r['texture_train']
    return r['hh_eligible'] and (r['cohort'] == 'primary' or version == 'v2')


def split(rows, fold_rows, fold, version):
    roles = {r['source'].replace('\\', '/'): r['role'] for r in fold_rows if int(r['fold']) == fold}
    assert len(roles) == 4000 and set(roles.values()) <= {'train', 'val', 'buffer'}
    tr, va = [], []
    for r in rows:
        valid_input = r['texture_input'] if version == 'v3_texture' else r['hh_eligible']
        if not valid_input:
            continue
        item = dict(r)
        if version == 'v3_texture':
            item.update(mask_area=r['texture_mask_area'], mask_status='texture_v2_original')
        if roles[r['source']] == 'train' and decides(r, version):
            tr.append(item)
        elif roles[r['source']] == 'val':
            va.append(item)
    assert all(r['mask_area'] > MINIMUM for r in tr+va)
    assert not ({r['source'] for r in tr} & {r['source'] for r in va})
    for cls in ('normal', 'underfill_50'):
        a = np.asarray([r['frame_index'] for r in tr if r['label'] == cls])
        b = np.asarray([r['frame_index'] for r in va if r['label'] == cls])
        assert a.size and b.size and np.min(np.abs(a[:, None]-b[None, :])) > 30
    return tr, va


def prepare_cache(args, rows):
    args.cache.mkdir(parents=True, exist_ok=True)
    config = dict(hh_manifest_sha=sha(args.bundle/'manifest.csv'),
        texture_manifest_sha=sha(args.texture/'texture_manifest.csv'), size=args.size,
        crop=[250, 0, 1250, 1000], resize='tensor bilinear antialias=True',
        minimum=args.rescue_min_area, source_order=[r['source'] for r in rows],
        adapter_sha=sha(__file__))
    marker = args.cache/'complete.json'
    shape = (len(rows), 3, args.size, args.size)
    if marker.exists():
        assert json.loads(marker.read_text()) == config, '새 설정은 새 캐시 경로를 사용하세요.'
        for name in ('raw', 'masked', 'texture'):
            a = np.load(args.cache/f'{name}.npy', mmap_mode='r')
            assert a.shape == shape and a.dtype == np.uint8
        return
    arrays = {name: np.lib.format.open_memmap(args.cache/f'{name}.npy', mode='w+',
              dtype=np.uint8, shape=shape) for name in ('raw', 'masked', 'texture')}
    for i, r in enumerate(rows):
        source = args.originals/r['source']
        assert sha(source) == r['source_sha256']
        with Image.open(source) as im:
            assert im.size == (1600, 1200)
            rgb = np.asarray(im.convert('RGB').crop((250, 0, 1250, 1000)))
        values = {'raw': rgb}
        for name, valid, root, path, digest, area in (
            ('masked', r['hh_eligible'], args.bundle, r['mask_path'], r['mask_sha256'], r['hh_mask_area']),
            ('texture', r['texture_input'], args.texture, r['texture_mask_path'], r['texture_mask_sha256'], r['texture_mask_area'])):
            if not valid:
                # 사용하지 않는 캐시 슬롯이다. 해당 사진은 이 버전의 학습/추론에 들어가지 않는다.
                values[name] = np.broadcast_to(np.array([124, 116, 104], np.uint8), rgb.shape).copy()
                continue
            mp = root/path
            assert sha(mp) == digest
            with Image.open(mp) as im:
                mask = np.asarray(im)
            assert mask.shape == (1000, 1000) and set(np.unique(mask)) <= {0, 255}
            assert abs(float((mask > 0).mean())-area) < 1e-12 and area > args.rescue_min_area
            value = rgb.copy()
            value[mask == 0] = (124, 116, 104)
            assert np.array_equal(value[mask > 0], rgb[mask > 0])
            values[name] = value
        for name, value in values.items():
            tensor = torch.from_numpy(value.copy()).permute(2, 0, 1)
            arrays[name][i] = TF.resize(tensor, [args.size, args.size],
                interpolation=v2.InterpolationMode.BILINEAR, antialias=True).numpy()
        if (i+1) % 500 == 0:
            print(f'공통 입력 캐시 {i+1}/{len(rows)}', flush=True)
    for a in arrays.values():
        a.flush()
    save_json(marker, config)


class Inputs:
    def __init__(self, cache, rows, version):
        name = {'v1': 'raw', 'v2': 'masked', 'v3_texture': 'texture'}[version]
        self.data = np.load(cache/f'{name}.npy', mmap_mode='r')
        self.index = {r['source']: i for i, r in enumerate(rows)}

    def __getitem__(self, source):
        return torch.from_numpy(self.data[self.index[source]].copy())


engine.split, engine.decides, engine.Inputs = split, decides, Inputs


def metric_record(version, seed, cohort, group, complete):
    return dict(version=version, seed=seed, cohort=cohort, all_folds_complete=complete,
        **engine.original.binary_metrics([r['label_id'] for r in group],
                                         [r['probability_underfill'] for r in group]))


def aggregate(output):
    oof = []
    completed = set()
    for p in sorted(output.glob('v*/seed*/fold*/complete.json')):
        cfg = json.loads(p.read_text())['config']
        assert cfg['version'] in VERSIONS
        completed.add((cfg['version'], cfg['seed'], cfg['fold']))
        oof.extend(read_csv(p.parent/'predictions.csv'))
    for r in oof:
        for k in ('seed', 'fold', 'label_id', 'pred', 'pipeline_decides', 'frame_index'):
            r[k] = int(r[k])
        r['probability_underfill'] = float(r['probability_underfill'])
        assert math.isfinite(r['probability_underfill']) and 0 <= r['probability_underfill'] <= 1
        assert r['pred'] == int(r['probability_underfill'] >= .5)
        assert float(r['mask_area']) > MINIMUM
    assert len(oof) == len({(r['version'], r['seed'], r['source']) for r in oof})
    write_csv(output/'oof_predictions.csv', oof)
    summary, two, three, pairs = [], [], [], []
    finished_seeds = []
    for seed in sorted({r['seed'] for r in oof}):
        maps = {v: {r['source']: r for r in oof if r['version'] == v and r['seed'] == seed} for v in VERSIONS}
        complete = {v: all((v, seed, f) in completed for f in range(5)) for v in VERSIONS}
        for v in VERSIONS:
            for cohort in ('all_available', 'primary', 'rescue', 'texture_only', 'pipeline_accepted'):
                group = [r for r in maps[v].values() if cohort == 'all_available' or r['cohort'] == cohort
                         or (cohort == 'pipeline_accepted' and r['pipeline_decides'])]
                if group:
                    summary.append(metric_record(v, seed, cohort, group, complete[v]))
        # 공통 사진만 비교한다. 미완료 fold끼리의 숫자를 완성된 비교로 내지 않는다.
        if complete['v1'] and complete['v2']:
            assert maps['v1'].keys() == maps['v2'].keys()
            for cohort in ('all_common_two', 'primary', 'rescue'):
                ids = sorted(k for k in maps['v1'] if cohort == 'all_common_two' or maps['v1'][k]['cohort'] == cohort)
                for v in ('v1', 'v2'):
                    two.append(metric_record(v, seed, cohort, [maps[v][k] for k in ids], True))
        if not all(complete.values()):
            continue
        finished_seeds.append(seed)
        common = set.intersection(*(set(maps[v]) for v in VERSIONS))
        for cohort in ('all_common_three', 'primary', 'rescue'):
            ids = sorted(k for k in common if cohort == 'all_common_three' or maps['v1'][k]['cohort'] == cohort)
            if not ids:
                continue
            for v in VERSIONS:
                three.append(metric_record(v, seed, cohort, [maps[v][k] for k in ids], True))
            for a, b in itertools.combinations(VERSIONS, 2):
                gains = losses = 0
                for k in ids:
                    x, y = maps[a][k], maps[b][k]
                    assert x['fold'] == y['fold'] and x['label_id'] == y['label_id']
                    cx, cy = x['pred'] == x['label_id'], y['pred'] == y['label_id']
                    gains += cy and not cx
                    losses += cx and not cy
                pairs.append(dict(seed=seed, a=a, b=b, cohort=cohort, n=len(ids),
                    b_only_correct=gains, a_only_correct=losses, accuracy_delta_b_minus_a=(gains-losses)/len(ids)))
    for name, data in (('summary.csv', summary), ('comparison_two.csv', two),
                        ('comparison_three.csv', three), ('paired_comparison.csv', pairs)):
        write_csv(output/name, data)
    means = []
    for scope, data in (('two', two), ('three', three)):
        for version, cohort in sorted({(r['version'], r['cohort']) for r in data}):
            group = [r for r in data if r['version'] == version and r['cohort'] == cohort]
            for metric in ('accuracy', 'balanced_accuracy', 'underfill_recall', 'normal_recall', 'auroc', 'fp', 'fn'):
                values = [r[metric] for r in group if r[metric] is not None]
                if values:
                    means.append(dict(scope=scope, version=version, cohort=cohort, metric=metric,
                        completed_seeds=','.join(str(r['seed']) for r in group), seed_count=len(values),
                        n_per_seed=group[0]['n'], mean=statistics.mean(values),
                        seed_sd=statistics.stdev(values) if len(values) > 1 else None))
    write_csv(output/'across_seeds.csv', means)
    save_json(output/'progress.json', dict(fully_completed_seeds=finished_seeds,
        completed_runs=len(completed), planned_runs=45, mean_sd_note='시드 간 표준편차이며 일반화 신뢰구간이 아님'))


def audit(args, rows, full, folds):
    records, coverage = [], []
    for version in VERSIONS:
        for fold in range(5):
            tr, va = split(rows, folds, fold, version)
            ref, _ = split(rows, folds, fold, 'v1')
            records.append(dict(version=version, fold=fold, train_n=len(tr), val_n=len(va),
                train_min_area=min(r['mask_area'] for r in tr), val_min_area=min(r['mask_area'] for r in va),
                buffer_frames=30, per_round_steps=math.ceil(len(ref)/args.batch),
                total_update_steps=math.ceil(len(ref)/args.batch)*args.rounds))
        for cls in ('normal', 'underfill_50'):
            accepted = {r['source'] for r in rows if r['label'] == cls and decides(r, version)}
            seq = sorted((r for r in full if r['label'] == cls), key=lambda r: int(r['frame_index']))
            held = best = current = 0
            for r in seq:
                ok = r['source'] in accepted
                held += not ok
                current = 0 if ok else current+1
                best = max(best, current)
            coverage.append(dict(version=version, label=cls, total=len(seq), accepted=len(accepted),
                held=held, coverage=len(accepted)/len(seq), longest_hold_run=best))
    write_csv(args.output/'eligibility_audit.csv', records)
    write_csv(args.output/'coverage.csv', coverage)
    shutil.copy2(args.bundle/'folds.csv', args.output/'folds.csv')
    shutil.copy2(args.texture/'preparation.json', args.output/'texture_preparation.json')
    exclusions = [r for r in full if r['original_cohort'] in {'primary', 'rescue'}
                  and not engine.eligible(r, args.rescue_min_area)]
    write_csv(args.output/'excluded_common_area.csv', exclusions)
    union = [dict(source=r['source'], label=r['label'], frame_index=r['frame_index'], cohort=r['cohort'],
        hh_eligible=int(r['hh_eligible']), texture_input=int(r['texture_input']), texture_train=int(r['texture_train']),
        hh_mask_area=r['hh_mask_area'], texture_mask_area=r['texture_mask_area'],
        texture_gate_ratio=r['texture_gate_ratio']) for r in rows]
    write_csv(args.output/'input_manifest.csv', union)
    save_json(args.output/'eligibility_policy.json', dict(minimum=args.rescue_min_area, comparison='>',
        common_excluded_count=len(exclusions), excluded_use='no training, validation or prediction',
        hh_input_count=sum(r['hh_eligible'] for r in rows),
        texture_train_count=sum(r['texture_train'] for r in rows),
        common_three_count=sum(r['hh_eligible'] and r['texture_input'] for r in rows),
        note='질감 게이트 비율과 확장된 실제 마스크 면적은 다른 수치다. 두 조건을 따로 기록한다.'))
    print(json.dumps(records, ensure_ascii=False), flush=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    rows, full = load_rows(args)
    folds = read_csv(args.bundle/'folds.csv')
    audit(args, rows, full, folds)
    protocol = dict(versions=list(VERSIONS), planned_seeds=[0, 1, 2], folds=[0, 1, 2, 3, 4],
        size=args.size, batch=args.batch, rounds=args.rounds, minimum=args.rescue_min_area,
        hh_manifest_sha=sha(args.bundle/'manifest.csv'), folds_sha=sha(args.bundle/'folds.csv'),
        texture_manifest_sha=sha(args.texture/'texture_manifest.csv'), adapter_sha=sha(__file__),
        engine_sha=sha(engine.__file__), equal_budget='same fold update count referenced to primary v1',
        final_selection='fixed last round', threshold=.5,
        limitations=['one print per class', 'not an independent test', 'offline rescue masks; online not validated',
                     'mask area shortcut unresolved', 'preprocessing changes training set composition'])
    pp = args.output/'run_settings.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol, '설정 변경: 새 RUN_NAME이 필요합니다.'
    else:
        save_json(pp, protocol)
    if args.verify_only:
        print('15가지 분할·30프레임 buffer·10% 초과 조건 점검 완료. 학습 없음.', flush=True)
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
                aggregate(args.output)
                # 완료 fold는 추론 가중치를 보관하고, 재개에 필요 없는 큰 optimizer 상태만 정리한다.
                folder = args.output/version/f'seed{seed}'/f'fold{fold}'
                if (folder/'complete.json').exists() and (folder/'model_state.pt').exists():
                    (folder/'latest.pt').unlink(missing_ok=True)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'texture', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--originals', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--rescue-min-area', type=float, default=MINIMUM)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if args.batch < 2 or args.size < 32 or args.rounds < 1 or not MINIMUM <= args.rescue_min_area < 1:
        p.error('batch>=2, size>=32, rounds>=1, 0.1<=minimum<1')
    if not set(args.seeds) <= {0, 1, 2} or not set(args.folds) <= set(range(5)):
        p.error('시드는 0/1/2, fold는 0..4를 사용하세요.')
    if not args.verify_only and (args.originals is None or args.cache is None):
        p.error('--originals와 --cache가 필요합니다.')
    main(args)
