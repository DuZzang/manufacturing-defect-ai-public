"""원본 DINOv2 OOF와 재활용 목록을 연결한다. 모델/마스크를 바꾸지 않는다."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def auc(y, p):
    n0, n1 = sum(y == 0), sum(y == 1)
    if not n0 or not n1:
        return None
    order = np.argsort(p)
    ys, ps = y[order], p[order]
    start, ranks = 0, 0.
    while start < len(ps):
        stop = start + 1
        while stop < len(ps) and ps[stop] == ps[start]:
            stop += 1
        ranks += ((start + 1 + stop) / 2.) * sum(ys[start:stop] == 1)
        start = stop
    return float((ranks - n1 * (n1 + 1) / 2) / (n0 * n1))


def measures(rows, total):
    normal_total = sum(int(r['label_id']) == 0 for r in total)
    under_total = len(total) - normal_total
    y = np.array([int(r['label_id']) for r in rows])
    p = np.array([float(r['probability_underfill']) for r in rows])
    pred = p >= .5
    tn, fp, fn, tp = (int(sum((y == a) & (pred == b))) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    return dict(total=len(total), judged=len(rows), held=len(total)-len(rows),
                normal_total=normal_total, underfill_total=under_total,
                normal_judged=tn+fp, underfill_judged=tp+fn,
                normal_held=normal_total-tn-fp, underfill_held=under_total-tp-fn,
                normal_coverage=(tn+fp)/normal_total if normal_total else None,
                underfill_coverage=(tp+fn)/under_total if under_total else None,
                tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=(tn+tp)/len(rows) if len(rows) else None,
                normal_recall=tn/(tn+fp) if tn+fp else None,
                underfill_recall=tp/(tp+fn) if tp+fn else None,
                auroc=auc(y, p))


def evaluate(out, rescue_manifest, source_manifest, folds_path):
    out = Path(out)
    rescue = read(rescue_manifest)
    assert len(rescue) == len({r['source'] for r in rescue}) == 743
    assert Counter(r['label'] for r in rescue) == {'normal': 379, 'underfill_50': 364}
    assert all(r['cohort'] == 'rescue' and float(r['mask_area']) >= .1 for r in rescue)
    manifest = {r['source']: r for r in read(source_manifest)}
    folds = {r['source'].replace('\\', '/'): int(r['fold']) for r in read(folds_path) if r['role'] == 'val'}
    predictions = read(out / 'oof_predictions_full_precision.csv')
    index = {}
    for r in predictions:
        s = r['source']
        assert s in manifest and int(r['fold']) == folds[s]
        assert r['label_id'] == manifest[s]['label_id']
        key = (r['condition'], s)
        assert key not in index
        index[key] = r
        if r['probability_underfill']:
            p = float(r['probability_underfill'])
            assert np.isfinite(p) and 0 <= p <= 1
            assert int(r['pred']) == int(p >= .5)
    assert len(index) == 16000
    conditions = sorted({r['condition'] for r in predictions})
    assert len(conditions) == 4
    common = [r for r in rescue if all(index[c, r['source']]['probability_underfill'] for c in conditions)]
    common_sources = {r['source'] for r in common}
    summary, fold_metrics, detailed, paired = [], [], [], []
    for condition in conditions:
        for scope, population in (('rescue743_all_candidates', rescue), ('rescue_common_all4', common)):
            candidates = [index[condition, r['source']] for r in population]
            judged = [r for r in candidates if r['probability_underfill']]
            summary.append(dict(condition=condition, scope=scope, **measures(judged, population)))
            for k in range(5):
                target = [r for r in population if folds[r['source']] == k]
                part = [r for r in judged if int(r['fold']) == k]
                fold_metrics.append(dict(condition=condition, scope=scope, fold=k, **measures(part, target)))
        for r in rescue:
            prediction = index[condition, r['source']]
            detailed.append(dict(prediction, shape_v2_area=r['mask_area'],
                                 common_all4=int(r['source'] in common_sources)))
    for left, right in zip(conditions[::2], conditions[1::2]):
        # 조건 정렬은 A,B,C,D. A/B는 공통743, C/D는 양쪽 통과 집합.
        population = [r for r in rescue if index[left,r['source']]['probability_underfill'] and index[right,r['source']]['probability_underfill']]
        gained = lost = 0
        for r in population:
            a, b = index[left,r['source']], index[right,r['source']]
            before, after = a['pred'] == a['label_id'], b['pred'] == b['label_id']
            gained += int(not before and after)
            lost += int(before and not after)
        paired.append(dict(before=left, after=right, common_n=len(population), corrected=gained,
                           new_errors=lost, net_correct_gain=gained-lost))
    write(out / 'rescue_summary.csv', summary)
    write(out / 'rescue_per_fold.csv', fold_metrics)
    write(out / 'rescue_predictions.csv', detailed)
    write(out / 'rescue_paired_changes.csv', paired)
    write(out / 'rescue_errors.csv', [r for r in detailed if r['probability_underfill'] and r['pred'] != r['label_id']])
    return summary
