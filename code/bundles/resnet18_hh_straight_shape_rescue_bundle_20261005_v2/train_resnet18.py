"""ResNet18 1/1.5/2 비교. 준비된 마스크 PNG만 읽고 생성/재선별하지 않는다.

v1=기존 통과 원본, v1_5=같은 통과 현환 베드지도 마스킹, v2=v1_5+형상v2 재활용.
고정 optimizer 업데이트 예산, 같은 평가 사진, 같은 시작 가중치. 본 학습은 Colab.
2026-10-05 Claude 다듬기: 재활용 중 남는 출력물 면적 < --rescue-min-area(기본 0.1, 질감 게이트 통과 기준과 같음)는
cohort 'rescue_small'로 바꿔 v2 학습·판정에서 빼고 진단 평가만 한다(면적 5% 미만 12장 중 10장이 정상이라 "거의 회색 → 정상"
지름길 위험). 0을 주면 원래 동작. 집계에 연속 보류 최장 길이(holds.csv)와 5장 다수결 정확도(vote.csv)를 추가.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

import baseline_classifier as original


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def write_csv(p,rows):
    if not rows:return
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save_json(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def load_rows(bundle,rescue_min_area=0.):
    rows=[r for r in read_csv(bundle/'manifest.csv') if r['cohort']!='held']
    for r in rows:
        r['label_id']=int(r['label_id']);r['frame_index']=int(r['frame_index']);r['mask_area']=float(r['mask_area'])
        if r['cohort']=='rescue' and r['mask_area']<rescue_min_area:r['cohort']='rescue_small'   # 평가는 하되 v2도 학습·판정 안 함
    assert len(rows)==3456 and len({r['source'] for r in rows})==3456
    return rows


def decides(r,version):return r['cohort']=='primary' or (version=='v2' and r['cohort']=='rescue')


def split(rows,fold_rows,fold,version):
    roles={r['source'].replace('\\','/'):r['role'] for r in fold_rows if int(r['fold'])==fold}
    assert len(roles)==4000
    tr=[r for r in rows if roles[r['source']]=='train' and decides(r,version)]
    va=[r for r in rows if roles[r['source']]=='val']
    assert not ({r['source'] for r in tr}&{r['source'] for r in va})
    for cls in ('normal','underfill_50'):
        a=np.array([r['frame_index'] for r in tr if r['label']==cls])
        b=np.array([r['frame_index'] for r in va if r['label']==cls])
        assert len(a)>0 and len(b)>0 and np.min(np.abs(a[:,None]-b[None,:]))>30
    return tr,va


def prepare_cache(args,rows):
    cache=args.cache;cache.mkdir(parents=True,exist_ok=True)
    cfg=dict(manifest=sha(args.bundle/'manifest.csv'),size=args.size,resize='tensor bilinear antialias=True',crop=[250,0,1250,1000])
    complete=cache/'complete.json'
    if complete.exists():
        assert json.loads(complete.read_text())==cfg,'캐시 설정 변경: 새 cache 경로 필요'
        for rep in ('raw','masked'):
            array=np.load(cache/f'{rep}.npy',mmap_mode='r')
            assert array.shape==(len(rows),3,args.size,args.size) and array.dtype==np.uint8
        return
    arrays={rep:np.lib.format.open_memmap(cache/f'{rep}.npy',mode='w+',dtype=np.uint8,
            shape=(len(rows),3,args.size,args.size)) for rep in ('raw','masked')}
    for i,r in enumerate(rows):
        source=args.originals/r['source'];mask_path=args.bundle/r['mask_path']
        assert sha(source)==r['source_sha256'] and sha(mask_path)==r['mask_sha256']
        with Image.open(source) as im:
            assert im.size==(1600,1200)
            rgb=np.array(im.convert('RGB').crop((250,0,1250,1000)))
        mask=np.array(Image.open(mask_path))
        assert mask.shape==(1000,1000) and set(np.unique(mask))<={0,255}
        masked=rgb.copy();masked[mask==0]=(124,116,104)
        assert np.array_equal(masked[mask>0],rgb[mask>0])
        for rep,value in (('raw',rgb),('masked',masked)):
            t=torch.from_numpy(value).permute(2,0,1)
            arrays[rep][i]=TF.resize(t,[args.size,args.size],interpolation=v2.InterpolationMode.BILINEAR,antialias=True).numpy()
        if (i+1)%500==0:print(f'입력 {i+1}/{len(rows)}',flush=True)
    for array in arrays.values():array.flush()
    save_json(complete,cfg)


class Inputs:
    def __init__(self,cache,rows,version):
        self.data=np.load(cache/('raw.npy' if version=='v1' else 'masked.npy'),mmap_mode='r')
        self.index={r['source']:i for i,r in enumerate(rows)}
    def __getitem__(self,k):return torch.from_numpy(self.data[self.index[k]].copy())


def predict(model,va,inputs,args,version,seed,fold):
    p,loss=original.predict(model,va,inputs,'source',torch.device('cuda'),args.batch,True)
    result=[dict(version=version,seed=seed,fold=fold,source=r['source'],label=r['label'],label_id=r['label_id'],
        frame_index=r['frame_index'],cohort=r['cohort'],mask_status=r['mask_status'],mask_area=r['mask_area'],
        probability_underfill=x,pred=int(x>=.5),pipeline_decides=int(decides(r,version))) for r,x in zip(va,p)]
    return result,loss


def run(args,rows,fold_rows,version,seed,fold):
    train,val=split(rows,fold_rows,fold,version)
    reference,_=split(rows,fold_rows,fold,'v1')
    per_round=math.ceil(len(reference)/args.batch);steps=per_round*args.rounds
    out=args.output/version/f'seed{seed}'/f'fold{fold}';out.mkdir(parents=True,exist_ok=True)
    config=dict(version=version,seed=seed,fold=fold,size=args.size,batch=args.batch,rounds=args.rounds,
        train_n=len(train),val_n=len(val),per_round_steps=per_round,total_steps=steps,
        manifest_sha=sha(args.bundle/'manifest.csv'),folds_sha=sha(args.bundle/'folds.csv'),
        trainer_sha=sha(__file__),baseline_sha=sha(args.bundle/'baseline_classifier.py'),
        torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,fast=True,
        lr_backbone=1e-4,lr_head=1e-3,weight_decay=1e-4,augmentation=original.AUGMENT_DESC,
        final_selection='fixed_last_round',rescue_min_area=args.rescue_min_area)
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config,'설정/환경 변경: 새 결과 폴더 필요'
    else:save_json(out/'config.json',config)
    if (out/'complete.json').exists():print(f'이미 완료 {version} seed{seed} fold{fold}',flush=True);return
    original.set_seed(seed*100+fold)
    model,head,_=original.build_model('resnet18');model.cuda()
    optimizer=torch.optim.AdamW([dict(params=[p for name,p in model.named_parameters() if not name.startswith('fc.')],lr=1e-4),
                                 dict(params=head.parameters(),lr=1e-3)],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps)
    scaler=torch.amp.GradScaler('cuda')
    aug=v2.Compose([v2.RandomHorizontalFlip(.5),v2.ColorJitter(brightness=.1,contrast=.1)])
    inputs=Inputs(args.cache,rows,version)
    generator=torch.Generator().manual_seed(seed*1000+fold)
    queue=[];history=[];seen=set();start=1;previous_seconds=0
    checkpoint=out/'latest.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False) # 자기 실행 체크포인트만 사용
        assert state['config']==config
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        generator.set_state(state['order_rng']);torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng'])
        queue=state['queue'];history=state['history'];seen=set(state['seen'])
        start=state['round']+1;previous_seconds=state['seconds']
    begin=time.perf_counter()
    for round_no in range(start,args.rounds+1):
        model.train();loss_sum=0
        for _ in range(per_round):
            while len(queue)<args.batch:queue.extend(torch.randperm(len(train),generator=generator).tolist())
            ix,queue=queue[:args.batch],queue[args.batch:];seen.update(ix)
            batch=[train[i] for i in ix]
            x=original.to_input(torch.stack([inputs[r['source']] for r in batch]),torch.device('cuda'),aug)
            y=torch.tensor([r['label_id'] for r in batch],device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(model(x),y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();scheduler.step()
            loss_sum+=float(loss.item())
        predictions,val_loss=predict(model,val,inputs,args,version,seed,fold)
        metric=original.binary_metrics([r['label_id'] for r in predictions],[r['probability_underfill'] for r in predictions])
        history.append(dict(round=round_no,train_loss=loss_sum/per_round,val_loss=val_loss,common_accuracy=metric['accuracy']))
        write_csv(out/'history.csv',history);write_csv(out/'predictions.csv',predictions)
        payload=dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),order_rng=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),queue=queue,history=history,seen=sorted(seen),round=round_no,
            seconds=previous_seconds+time.perf_counter()-begin)
        torch.save(payload,out/'latest.tmp.pt');os.replace(out/'latest.tmp.pt',checkpoint)
        print(f'{version} seed{seed} fold{fold} round {round_no}/{args.rounds}: loss {loss_sum/per_round:.4f}, common acc {metric["accuracy"]:.4f}',flush=True)
    torch.save(model.state_dict(),out/'model_state.pt')
    save_json(out/'complete.json',dict(config=config,distinct_training_images_seen=len(seen),
        effective_sample_passes=steps*args.batch/len(train),seconds=previous_seconds+time.perf_counter()-begin))


def aggregate(output):
    oof=[]
    for p in output.glob('v*/seed*/fold*/complete.json'):oof.extend(read_csv(p.parent/'predictions.csv'))
    for r in oof:
        for k in ('seed','fold','label_id','pred','pipeline_decides','frame_index'):r[k]=int(r[k])
        r['probability_underfill']=float(r['probability_underfill'])
    summary=[]
    for version,seed in sorted({(r['version'],r['seed']) for r in oof}):
        predictions=[r for r in oof if r['version']==version and r['seed']==seed]
        assert len(predictions)==len({r['source'] for r in predictions})
        folds=sorted({r['fold'] for r in predictions})
        for cohort in ('all_common','primary','rescue','rescue_small','pipeline_accepted'):
            group=[r for r in predictions if cohort=='all_common' or r['cohort']==cohort or cohort=='pipeline_accepted' and r['pipeline_decides']]
            if not group:continue
            metrics=original.binary_metrics([r['label_id'] for r in group],[r['probability_underfill'] for r in group])
            summary.append(dict(version=version,seed=seed,cohort=cohort,completed_folds=','.join(map(str,folds)),
                                all_folds_complete=folds==[0,1,2,3,4],**metrics))
    write_csv(output/'summary.csv',summary);write_csv(output/'oof_predictions.csv',oof)
    pairs=[]
    for seed in sorted({r['seed'] for r in oof}):
        for a,b in (('v1','v1_5'),('v1_5','v2')):
            left={r['source']:r for r in oof if r['seed']==seed and r['version']==a}
            right={r['source']:r for r in oof if r['seed']==seed and r['version']==b}
            for cohort in ('all_common','primary','rescue'):
                ids=[k for k in left.keys()&right.keys() if cohort=='all_common' or left[k]['cohort']==cohort]
                gains=losses=0
                for k in ids:
                    x,y=left[k],right[k];assert x['fold']==y['fold'] and x['label_id']==y['label_id']
                    cx,cy=x['pred']==x['label_id'],y['pred']==y['label_id'];gains+=cy and not cx;losses+=cx and not cy
                pairs.append(dict(seed=seed,a=a,b=b,cohort=cohort,n=len(ids),b_only_correct=gains,a_only_correct=losses,
                    accuracy_delta_b_minus_a=(gains-losses)/len(ids) if ids else None))
    write_csv(output/'paired_comparison.csv',pairs)
    # 5장 다수결: 판정하는 사진만 촬영 순서로(fold·클래스별), 보류 사진은 건너뜀. pipeline.py 실시간 방식과 같음
    vote=[]
    for version,seed in sorted({(r['version'],r['seed']) for r in oof}):
        hit=n=0
        for key in {(r['fold'],r['label_id']) for r in oof}:
            seq=sorted((r for r in oof if r['version']==version and r['seed']==seed and (r['fold'],r['label_id'])==key and r['pipeline_decides']),key=lambda r:r['frame_index'])
            for i in range(2,len(seq)):
                w=seq[max(0,i-4):i+1];n+=1;hit+=(sum(r['pred'] for r in w)*2>len(w))==bool(key[1])
        vote.append(dict(version=version,seed=seed,window=5,min_votes=3,decided=n,vote_accuracy=hit/n if n else None))
    write_csv(output/'vote.csv',vote)


def hold_runs(bundle,rows,output):
    """모델과 무관: 버전별로 판정하지 않는(보류) 연속 프레임의 최장 길이. 4장/초 가정이면 /4 초."""
    status={r['source']:r for r in rows};out=[]
    for version in ('v1','v1_5','v2'):
        for cls in ('normal','underfill_50'):
            frames=sorted((r for r in read_csv(bundle/'manifest.csv') if r['label']==cls),key=lambda r:int(r['frame_index']))
            best=cur=held=0
            for r in frames:
                ok=r['source'] in status and decides(status[r['source']],version)
                cur=0 if ok else cur+1;best=max(best,cur);held+=not ok
            out.append(dict(version=version,label=cls,frames=len(frames),held=held,hold_rate=held/len(frames),longest_hold_run=best))
    write_csv(output/'holds.csv',out);return out


def main(args):
    rows=load_rows(args.bundle,args.rescue_min_area);fold_rows=read_csv(args.bundle/'folds.csv')
    for fold in range(5):
        a,av=split(rows,fold_rows,fold,'v1');m,mv=split(rows,fold_rows,fold,'v1_5');b,bv=split(rows,fold_rows,fold,'v2')
        assert a==m and av==mv==bv and {r['source'] for r in a}<={r['source'] for r in b}
    small=[r['label'] for r in rows if r['cohort']=='rescue_small']
    print(f"재활용 면적<{args.rescue_min_area}: {len(small)}장 진단만 (정상 {small.count('normal')}, 언더필 {small.count('underfill_50')})",flush=True)
    if args.verify_only:print('목록/5fold/30frame buffer 검증 완료. 학습 없음.',flush=True);return
    prepare_cache(args,rows)
    if args.prepare_only:print('입력 캐시 완료. 학습 없음.',flush=True);return
    assert torch.cuda.is_available(),'Colab GPU에서 실행해야 함'
    args.output.mkdir(parents=True,exist_ok=True)
    torch.backends.cudnn.benchmark=True
    save_json(args.output/'environment.json',dict(torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,
        gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,fast_nondeterministic=True,main_training='Colab'))
    for name in ('preparation.json','coverage.csv'):__import__('shutil').copy2(args.bundle/name,args.output/name)
    save_json(args.output/'run_settings.json',dict(rescue_min_area=args.rescue_min_area,rounds=args.rounds,size=args.size,batch=args.batch))
    for h in hold_runs(args.bundle,rows,args.output):print(f"보류 {h['version']} {h['label']}: {h['held']}장, 최장 연속 {h['longest_hold_run']}장",flush=True)
    for seed in args.seeds:
        for fold in args.folds:
            for version in ('v1','v1_5','v2'):
                run(args,rows,fold_rows,version,seed,fold);aggregate(args.output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--originals',type=Path)
    p.add_argument('--cache',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--size',type=int,default=512);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--rounds',type=int,default=10);p.add_argument('--seeds',type=int,nargs='+',default=[0])
    p.add_argument('--folds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--rescue-min-area',type=float,default=.1,help='재활용 사진의 최소 출력물 면적. 0이면 804장 전부 사용')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.batch<2 or args.size<32 or args.rounds<1:p.error('batch>=2, size>=32, rounds>=1')
    if not args.verify_only and (args.originals is None or args.cache is None):p.error('--originals --cache 필요')
    if not (args.verify_only or args.prepare_only) and args.output is None:p.error('--output 필요')
    main(args)
