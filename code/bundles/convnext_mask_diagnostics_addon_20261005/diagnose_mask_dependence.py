"""기존 ConvNeXt 가중치의 마스크 민감도 점검. 학습·가중치 갱신 없음."""
import argparse
import collections
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

import train_convnext_three as trainer

VERSIONS = ('v2', 'v3_texture')
ARMS = ('baseline', 'outside_black', 'outside_white', 'mask_only')
GRAY = (124, 116, 104)


def area_bin(area):
    return '10-30%' if area < .3 else '30-60%' if area < .6 else '60-90%' if area < .9 else '90-100%'


def select_sample(items, maximum, seed):
    """정답/오답·예측 확률을 사용하지 않는 고정 층화 표본."""
    buckets = collections.defaultdict(list)
    for item in items:
        key = (item['label'], item['fold'], item['cohort'], area_bin(item['mask_area']))
        buckets[key].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['source']}".encode()).hexdigest())
    result = []
    keys = sorted(buckets)
    depth = 0
    while len(result) < min(maximum, len(items)):
        for key in keys:
            if depth < len(buckets[key]):
                result.append(buckets[key][depth])
                if len(result) == min(maximum, len(items)):
                    break
        depth += 1
    return result


def mask_coverage(mask, size):
    assert mask.ndim == 2 and set(np.unique(mask)) <= {0, 255}
    source = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
    return TF.resize(source, [size, size], interpolation=v2.InterpolationMode.BILINEAR,
                     antialias=True)[0].numpy()


def make_variants(base, coverage):
    """축소 후 원본이 조금이라도 기여하는 화소/경계는 색상 검사에서 보호."""
    assert base.dtype == np.uint8 and base.shape == (3, *coverage.shape)
    safe_outside = coverage == 0
    variants = {'baseline': base.copy()}
    for name, color in (('outside_black', (0, 0, 0)), ('outside_white', (255, 255, 255))):
        image = base.copy()
        image[:, safe_outside] = np.asarray(color, np.uint8)[:, None]
        assert np.array_equal(image[:, ~safe_outside], base[:, ~safe_outside])
        variants[name] = image
    # 실제 RGB는 전부 지운다. 모양만 남기는 OOD 보조 검사이며 지름길의 증명은 아니다.
    shape = np.broadcast_to(np.asarray(GRAY, np.uint8)[:, None, None], base.shape).copy()
    shape[:, ~safe_outside] = 224
    variants['mask_only'] = shape
    return variants, float(safe_outside.mean())


@torch.inference_mode()
def infer(model, array, device, batch):
    result = []
    for start in range(0, len(array), batch):
        x = trainer.engine.original.to_input(torch.from_numpy(np.stack(array[start:start+batch])), device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(x)
        result.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
    assert all(np.isfinite(x) and 0 <= x <= 1 for x in result)
    return result


def correlation(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(items):
    output = []
    for version, seed in sorted({(r['version'], r['seed']) for r in items}):
        subset = [r for r in items if (r['version'], r['seed']) == (version, seed)]
        for scope in ('all_sampled', 'changeable_outside', 'primary', 'rescue', 'normal', 'underfill_50'):
            group = [r for r in subset if scope == 'all_sampled'
                     or scope == 'changeable_outside' and r['changed_fraction'] > 0
                     or scope == r['cohort'] or scope == r['label']]
            if not group:
                continue
            labels = [r['label_id'] for r in group]
            for arm in ARMS:
                probs = [r['prob_'+arm] for r in group]
                deltas = np.asarray([p-r['prob_baseline'] for p, r in zip(probs, group)])
                flips = sum((p >= .5) != (r['prob_baseline'] >= .5) for p, r in zip(probs, group))
                output.append(dict(version=version, seed=seed, scope=scope, arm=arm,
                    interpretation='OOD auxiliary only' if arm == 'mask_only' else 'fill-color sensitivity',
                    **trainer.engine.original.binary_metrics(labels, probs),
                    flips=flips, flip_rate=flips/len(group), mean_abs_probability_change=float(np.abs(deltas).mean()),
                    p95_abs_probability_change=float(np.quantile(np.abs(deltas), .95)),
                    mean_signed_probability_change=float(deltas.mean()),
                    max_abs_probability_change=float(np.abs(deltas).max())))
    return output


def area_relations(items):
    """전체 OOF에서 관계를 기술. 면적이 원인이라는 결론으로 해석하지 않는다."""
    result = []
    for version, seed in sorted({(r['version'], r['seed']) for r in items}):
        subset = [r for r in items if (r['version'], r['seed']) == (version, seed)]
        for scope in ('all_oof', 'normal', 'underfill_50', 'primary', 'rescue', *[f'fold{i}' for i in range(5)]):
            group = [r for r in subset if scope == 'all_oof' or r['label'] == scope
                     or r['cohort'] == scope or f"fold{r['fold']}" == scope]
            if not group:
                continue
            area = [r['mask_area'] for r in group]
            probs = [r['saved_probability'] for r in group]
            labels = [r['label_id'] for r in group]
            result.append(dict(version=version, seed=seed, scope=scope, n=len(group),
                normal_n=labels.count(0), underfill_n=labels.count(1),
                area_only_auroc=trainer.engine.original.auroc(labels, area),
                area_probability_pearson=correlation(area, probs),
                caveat='Association only; class, phase, lighting and defect evidence may confound it.'))
    return result


def load_checkpoint(path, device):
    # 저장된 전체 state_dict로 덮으므로 사전학습 가중치를 다시 다운로드하지 않는다.
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval()


def run(args):
    start = time.perf_counter()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows, _ = trainer.load_rows(args)
    folds = trainer.read_csv(args.bundle/'folds.csv')
    marker = json.loads((args.cache/'complete.json').read_text())
    assert marker['source_order'] == [r['source'] for r in rows]
    assert marker['size'] == args.size
    assert marker['hh_manifest_sha'] == trainer.sha(args.bundle/'manifest.csv')
    assert marker['texture_manifest_sha'] == trainer.sha(args.texture/'texture_manifest.csv')
    index = {r['source']: i for i, r in enumerate(rows)}
    arrays = {v: np.load(args.cache/f'{name}.npy', mmap_mode='r')
              for v, name in (('v2', 'masked'), ('v3_texture', 'texture'))}
    population, chosen, checkpoints = [], [], []
    for seed in args.seeds:
        for version in VERSIONS:
            candidates = []
            for fold in range(5):
                folder = args.results/version/f'seed{seed}'/f'fold{fold}'
                assert (folder/'complete.json').exists(), f'시드가 완료되지 않음: {folder}'
                cfg = json.loads((folder/'complete.json').read_text())['config']
                assert (cfg['version'], cfg['seed'], cfg['fold'], cfg['size']) == (version, seed, fold, args.size)
                assert cfg['manifest_sha'] == trainer.sha(args.bundle/'manifest.csv')
                assert cfg['folds_sha'] == trainer.sha(args.bundle/'folds.csv')
                assert cfg['texture_manifest_sha'] == trainer.sha(args.texture/'texture_manifest.csv')
                assert cfg['rescue_min_area'] == args.rescue_min_area
                assert cfg['adapter_sha'] == trainer.sha(trainer.__file__)
                assert cfg['trainer_sha'] == trainer.sha(trainer.engine.__file__)
                assert cfg['baseline_sha'] == trainer.sha(args.bundle/'baseline_classifier.py')
                checkpoint = folder/'model_state.pt'
                assert checkpoint.exists(), f'모델 가중치 없음: {checkpoint}. 결과 ZIP만으로는 검사할 수 없습니다.'
                checkpoints.append(dict(version=version, seed=seed, fold=fold,
                    path=str(checkpoint), sha256=trainer.sha(checkpoint)))
                _, val = trainer.split(rows, folds, fold, version)
                predictions = {r['source']: r for r in trainer.read_csv(folder/'predictions.csv')}
                assert set(predictions) == {r['source'] for r in val}
                for r in val:
                    old = predictions[r['source']]
                    assert int(old['label_id']) == r['label_id'] and int(old['fold']) == fold
                    candidates.append(dict(r, version=version, seed=seed, fold=fold,
                        saved_probability=float(old['probability_underfill'])))
            population.extend(candidates)
            chosen.extend(select_sample(candidates, args.max_per_version, seed))
    assert len(chosen) == len({(r['version'], r['seed'], r['source']) for r in chosen})
    settings = dict(training_run=False, optimizer_step_run=False, diagnostics_sha=trainer.sha(__file__),
        versions=list(VERSIONS), seeds=args.seeds, size=args.size, batch=args.batch, device=args.device,
        max_per_version=args.max_per_version, baseline_saved_probability_tolerance=.005,
        crop=[250, 0, 1250, 1000], minimum_area=args.rescue_min_area, comparison='>',
        threshold=.5, sampling='round-robin label/fold/cohort/area-bin; SHA order; no error/probability selection',
        changing='Only resized pixels with zero RGB-mask coverage; all evidence and interpolation boundary frozen.',
        mask_only='All real texture removed, valid support=224, invalid=RGB(124,116,104); OOD auxiliary.',
        checkpoints=checkpoints, data_bundle=str(args.bundle), cache=str(args.cache), results=str(args.results),
        source_notebook='ConvNeXt_three_preproc_20261005.ipynb',
        limitations=['No causal proof of shortcut; no new-print robustness verification.',
                    'Stratified sampled rates are not the prevalence-weighted whole-dataset performance.',
                    'Black/white and mask-only can be out of the training distribution.',
                    'No color effect does not rule out shape, area, phase or lighting shortcut.',
                    'v1 is intentionally excluded: removing its real bed would change image evidence.'])
    settings_file = output/'settings.json'
    if settings_file.exists():
        assert json.loads(settings_file.read_text()) == settings, '다른 설정이면 새 진단 output을 사용하세요.'
    trainer.save_json(settings_file, settings)
    trainer.write_csv(output/'sample_manifest.csv', [dict(version=r['version'], seed=r['seed'], fold=r['fold'],
        source=r['source'], label=r['label'], cohort=r['cohort'], mask_area=r['mask_area'],
        area_bin=area_bin(r['mask_area'])) for r in chosen])
    trainer.write_csv(output/'area_relations_oof.csv', area_relations(population))
    if args.verify_only:
        print(f'가중치·OOF·입력·면적·표본 {len(chosen)}장 점검. 모델 추론/학습 없음.', flush=True)
        return
    device = torch.device(args.device)
    assert device.type != 'cuda' or torch.cuda.is_available(), 'GPU 연결을 확인하세요.'
    results = []
    baseline_tolerance = .005  # AMP·배치 크기 차이. 초과 시 재현 실패로 중단.
    for seed in args.seeds:
        for version in VERSIONS:
            for fold in range(5):
                group = [r for r in chosen if (r['seed'], r['version'], r['fold']) == (seed, version, fold)]
                if not group:
                    continue
                folder = args.results/version/f'seed{seed}'/f'fold{fold}'
                model = load_checkpoint(folder/'model_state.pt', device)
                inputs = {arm: [] for arm in ARMS}
                records = []
                for r in group:
                    path = args.texture/r['texture_mask_path'] if version == 'v3_texture' else args.bundle/r['mask_path']
                    expected = r['texture_mask_sha256'] if version == 'v3_texture' else r['mask_sha256']
                    assert trainer.sha(path) == expected
                    with Image.open(path) as im:
                        mask = np.asarray(im)
                    assert mask.shape == (1000, 1000)
                    assert abs(float((mask > 0).mean())-r['mask_area']) < 1e-12
                    assert r['mask_area'] > args.rescue_min_area
                    base = arrays[version][index[r['source']]].copy()
                    variants, changeable = make_variants(base, mask_coverage(mask, args.size))
                    for arm in ARMS:
                        inputs[arm].append(variants[arm])
                    records.append(dict(version=version, seed=seed, fold=fold, source=r['source'],
                        label=r['label'], label_id=r['label_id'], frame_index=r['frame_index'], cohort=r['cohort'],
                        mask_area=r['mask_area'], changed_fraction=changeable, saved_probability=r['saved_probability']))
                for arm in ARMS:
                    probs = infer(model, inputs[arm], device, args.batch)
                    for record, p in zip(records, probs):
                        record['prob_'+arm] = p
                for r in records:
                    error = abs(r['prob_baseline']-r['saved_probability'])
                    assert error <= baseline_tolerance, f'기존 baseline 재현 불일치: {r["source"]}: {error}'
                    r['baseline_saved_abs_error'] = error
                results.extend(records)
                # 실행 중 끊겨도 완료 fold의 진단 CSV는 Drive에 남는다.
                trainer.write_csv(output/'predictions.csv', results)
                trainer.write_csv(output/'summary.csv', summarize(results))
                print(f'점검 완료: {version} seed{seed} fold{fold}, {len(group)}장 (학습 없음)', flush=True)
                del model, inputs
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    ranked = sorted(results, key=lambda r: max(abs(r['prob_outside_black']-r['prob_baseline']),
                        abs(r['prob_outside_white']-r['prob_baseline'])), reverse=True)
    trainer.write_csv(output/'largest_changes.csv', ranked[:30])
    trainer.save_json(output/'complete.json', dict(training_run=False, weight_updates=0, n=len(results),
        seconds=time.perf_counter()-start, torch=str(torch.__version__),
        torchvision=str(__import__('torchvision').__version__), device=str(device),
        baseline_saved_max_abs_error=max(r['baseline_saved_abs_error'] for r in results),
        note='Not evidence that Layer Masking improves accuracy or generalization.'))
    print(f'진단 저장: {output}. 추가 학습 0회.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'texture', 'cache', 'results', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--max-per-version', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--rescue-min-area', type=float, default=.1)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.max_per_version < 1 or args.batch < 1 or args.size < 32 or not .1 <= args.rescue_min_area < 1:
        parser.error('양수 표본·배치, size>=32, minimum>=0.1이 필요합니다.')
    run(args)
