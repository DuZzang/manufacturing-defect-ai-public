#!/usr/bin/env python3
"""원본 RGB에서 베드와 출력물이 함께 보이는 50장(임시 후보)을 준비한다.

기존 fold0/4 검증·완충 구간과 겹치지 않는 언더필 학습 프레임만 이용한다.
기존 stage2 선별 통과 및 bedmap의 출력물 추정 영역 25~80%인 1000px 고정
크롭을 256px로 축소한다. 25~80%는 시각적 예비 선정값이며 정답이 아니다.
마스킹하지 않으며 원본 베드·표면을 그대로 남긴다. 품질·라벨 육안 검토 필요.
"""
import argparse, csv, hashlib, io, json, random, sys
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from prepare_data import InputStore, read_csv, partition, spread_sample

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--original-root'); group.add_argument('--original-zip')
    p.add_argument('--bundle',required=True); p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    out=Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'출력 폴더가 비어 있지 않음: {out}')
    original=InputStore(Path(args.original_root or args.original_zip)); bundle=InputStore(Path(args.bundle))
    try:
        folds_raw=bundle.read('folds.csv'); sel_raw=bundle.read('selections.csv')
        folds=partition(read_csv(bundle,'folds.csv')); selections=read_csv(bundle,'selections.csv')
        eligible=[]
        for row in selections:
            src=row['source'].replace('\\','/')
            if src not in folds: raise ValueError(f'원본 분할 없음: {src}')
            info=folds[src]
            if info['split']!='train' or row['label']!='underfill_50' or row['stage2']!='1': continue
            ratio=float(row['bedmap_area'])
            if 0.25<=ratio<=0.80:
                eligible.append(dict(source=src,frame_index=info['frame_index'],split=info['split'],
                    fold0_role=info['fold0_role'],fold4_role=info['fold4_role'],
                    stage2=row['stage2'],stage2_reason=row['stage2_reason'],surface_ratio_estimate=ratio,
                    mask_status=row.get('bedmap_status','')))
        if len(eligible)<50: raise ValueError(f'적격 후보 부족: {len(eligible)}장')
        selected=spread_sample(eligible,50,random.Random(args.seed))
        if len({r['source'] for r in selected})!=50: raise ValueError('선정 사진 중복')
        folder=out/'data'/'train_real'/'underfill'; folder.mkdir(parents=True)
        sheet=Image.new('RGB',(180*10,202*5),'white'); draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(selected):
            raw=original.read(row['source'])
            with Image.open(io.BytesIO(raw)) as im:
                im=ImageOps.exif_transpose(im).convert('RGB')
                if im.size!=(1600,1200): raise ValueError(f'원본 크기 상이 {im.size}: {row["source"]}')
                crop=im.crop((250,0,1250,1000)).resize((256,256),Image.Resampling.LANCZOS)
                dest=folder/(Path(row['source']).stem+'.png'); crop.save(dest)
                sheet.paste(crop.resize((180,180),Image.Resampling.LANCZOS),((i%10)*180,(i//10)*202))
                draw.text(((i%10)*180+3,(i//10)*202+182),f'{row["frame_index"]} r={row["surface_ratio_estimate"]:.2f}', fill='black')
            row.update(original_sha256=hashlib.sha256(raw).hexdigest(),crop_x=250,crop_y=0,
                       crop_size=1000,output_size=256,output_path=dest.relative_to(out).as_posix())
        fields=list(selected[0])
        with (out/'manifest_train.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(selected)
        sheet.save(out/'real_bed_and_surface_50.jpg',quality=92)
        config=dict(purpose='bed and underfill surface generation only; exploratory', count=len(selected),seed=args.seed,
            eligible_count=len(eligible),candidate_rule='train only; existing stage2 pass; bedmap estimated surface 25-80%',
            mask_note='estimated surface fraction, not ground truth; no masking of output pixels',
            crop_xyxy_original=[250,0,1250,1000],output_size=256,resize='Pillow LANCZOS',
            folds_sha256=hashlib.sha256(folds_raw).hexdigest(), selections_sha256=hashlib.sha256(sel_raw).hexdigest(),
            selection='seeded random selection from candidate-rank bins sorted by frame number',
            warnings=['single print condition; not independent samples', 'label indicates print condition, not visible defect ground truth',
                      '25-80 percent is a temporary heuristic; masks can fail', '1000px fixed crop resized to 256; fine detail may be lost',
                      'inspection of all samples required; stage2 may exclude certain conditions'])
        (out/'selection_settings.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'Eligible: {len(eligible)}; selected: {len(selected)}; output: {folder}',flush=True)
    finally:
        original.close(); bundle.close()
if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'준비 실패: {exc}',file=sys.stderr);raise
