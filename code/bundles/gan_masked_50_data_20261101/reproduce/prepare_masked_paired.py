#!/usr/bin/env python3
"""이전 베드 GAN의 실사 50장과 최대한 같은 사진으로 기존 v3 마스크를 적용한다.

참조 50장 중 v3에서 보류된 사진은 이전 50장 밖의 인접 프레임으로 교체한다.
기존 현환 직선 보정/형상 v2 마스크를 그대로 사용하며 게이트를 새로 적용하지 않는다.
"""
import argparse, csv, hashlib, io, json, sys, zipfile
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from prepare_data import InputStore, read_csv, partition


def digest(b): return hashlib.sha256(b).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--original-root'); source.add_argument('--original-zip')
    ap.add_argument('--v3-bundle',required=True)
    ap.add_argument('--fold-bundle',required=True)
    ap.add_argument('--reference-manifest',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'비어 있지 않은 결과 폴더: {out}')
    original=InputStore(Path(args.original_root or args.original_zip))
    v3=InputStore(Path(args.v3_bundle)); folds=InputStore(Path(args.fold_bundle))
    try:
        v3rows=read_csv(v3,'manifest.csv'); mapping={r['source'].replace('\\','/'):r for r in v3rows}
        split=partition(read_csv(folds,'folds.csv'))
        with Path(args.reference_manifest).open(encoding='utf-8-sig',newline='') as f: ref=list(csv.DictReader(f))
        if len(ref)!=50 or len({r['source'] for r in ref})!=50: raise ValueError('참조 목록에 고유 원본 50장이 필요합니다.')
        prior={r['source'] for r in ref}
        valid=lambda r: (r['label']=='underfill_50' and r['cohort'] in ('primary','rescue')
                           and bool(r['mask_path']) and float(r['mask_area'])>0.10
                           and r['source'] in split and split[r['source']]['split']=='train')
        selected=[]; missing=[]
        for old in ref:
            src=old['source']; row=mapping[src]
            if valid(row): selected.append((row,src,''))
            else: missing.append(old)
        used={r[0]['source'] for r in selected}
        candidates=[r for r in v3rows if valid(r) and r['source'] not in prior and 0.25<=float(r['mask_area'])<=0.80]
        replacements=[]
        for old in missing:
            avail=[r for r in candidates if r['source'] not in used]
            if not avail: raise ValueError('교체 후보가 부족합니다.')
            r=min(avail,key=lambda r:(abs(int(r['frame_index'])-int(old['frame_index'])),int(r['frame_index'])))
            used.add(r['source']); selected.append((r,old['source'],'nearest eligible frame')); replacements.append((old['source'],r['source']))
        selected.sort(key=lambda t:int(t[0]['frame_index']))
        if len(selected)!=50: raise ValueError(f'선정 수: {len(selected)}')
        folder=out/'data'/'train_real'/'underfill';folder.mkdir(parents=True)
        sheet=Image.new('RGB',(1800,202*5),'white'); draw=ImageDraw.Draw(sheet)
        manifest=[]
        for i,(row,ref_src,replacement) in enumerate(selected):
            raw=original.read(row['source']); mask_raw=v3.read(row['mask_path'])
            if row['source_sha256'] and row['source_sha256']!=digest(raw): raise ValueError('원본 해시 불일치 '+row['source'])
            if row['mask_sha256'] and row['mask_sha256']!=digest(mask_raw): raise ValueError('마스크 해시 불일치 '+row['mask_path'])
            with Image.open(io.BytesIO(raw)) as img, Image.open(io.BytesIO(mask_raw)) as mask:
                img=ImageOps.exif_transpose(img).convert('RGB'); mask=mask.convert('L')
                if img.size!=(1600,1200) or mask.size!=(1000,1000): raise ValueError('크기 불일치 '+row['source'])
                mask=mask.point(lambda p: 255 if p>0 else 0)
                actual_ratio=mask.histogram()[255]/1_000_000
                if actual_ratio<=0.10: raise ValueError('마스크 면적 10% 이하: '+row['source'])
                if abs(actual_ratio-float(row['mask_area']))>0.000002: raise ValueError('마스크 면적 불일치 '+row['source'])
                crop=img.crop((250,0,1250,1000)); background=Image.new('RGB',crop.size,(124,116,104))
                result=Image.composite(crop,background,mask).resize((256,256),Image.Resampling.LANCZOS)
                dest=folder/(Path(row['source']).stem+'.png');result.save(dest)
                x=(i%10)*180;y=(i//10)*202
                sheet.paste(result.resize((180,180),Image.Resampling.LANCZOS),(x,y))
                draw.text((x+2,y+182),f'{row["frame_index"]} {row["cohort"]} {actual_ratio:.2f}',fill='black')
            manifest.append(dict(source=row['source'],frame_index=row['frame_index'],label=row['label'],
                train_split=split[row['source']]['split'],fold0_role=split[row['source']]['fold0_role'],
                fold4_role=split[row['source']]['fold4_role'],cohort=row['cohort'],mask_path=row['mask_path'],
                mask_status=row['mask_status'],mask_origin=row['mask_origin'],mask_area_1000=actual_ratio,
                original_sha256=digest(raw),mask_sha256=digest(mask_raw),reference_source=ref_src,
                replacement_reason=replacement, crop_original_xyxy='250,0,1250,1000',output_size=256,
                gray_rgb='124,116,104',output_path=dest.relative_to(out).as_posix()))
        with (out/'manifest_train.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
        sheet.save(out/'real_masked_underfill_50.jpg',quality=92)
        settings=dict(count=50,paired_count=50-len(missing),replacement_count=len(missing),replacements=replacements,
          selection='Prior bed GAN frame list; v3 valid primary/rescue; replace held by nearest eligible train frame',
          mask_sources='Existing v3 primary HH bed-map straight-boundary and rescue shape v2',
          mask_rule='existing 1000x1000 binary mask >10%; no new gates or mask algorithm',gray_rgb=[124,116,104],
          crop_original_xyxy=[250,0,1250,1000],resized_output=[256,256],interpolation='Pillow LANCZOS',
          split='fold0 and fold4 holdouts and their 30-frame buffers excluded; existing time block',
          warnings=['masks are automatic estimates, not ground truth','some input pixels near mask boundaries get interpolation',
            'same 1 print per class, condition label not individual visual ground truth','4 substitutions affect paired comparison'])
        (out/'selection_settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(settings,ensure_ascii=False,indent=2),flush=True)
    finally:
        original.close();v3.close();folds.close()
if __name__=='__main__':
    try: main()
    except Exception as exc: print(f'준비 실패: {exc}',file=sys.stderr);raise
