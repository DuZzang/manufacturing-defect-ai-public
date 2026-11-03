"""ConvNeXt-Tiny: 기존 원본 크롭과 마스킹+추가 활용 비교. 면적 10% 초과만 사용."""
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
from area_policy import eligible, MIN_AREA


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def write_csv(p,rows):
    if not rows:return
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save_json(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')



def load_rows(bundle,rescue_min_area=MIN_AREA):
    records=read_csv(bundle/'manifest.csv')
    rows=[]
    for r in records:
        if not eligible(r,rescue_min_area):continue
        r['label_id']=int(r['label_id']);r['frame_index']=int(r['frame_index']);r['mask_area']=float(r['mask_area'])
        rows.append(r)
    assert rows and len(rows)==len({r['source'] for r in rows})
    assert all(eligible(r,rescue_min_area) for r in rows)
    return rows


def decides(r,version):return r['cohort']=='primary' or (version=='v2' and r['cohort']=='rescue')


def split(rows,fold_rows,fold,version):
    roles={r['source'].replace('\\','/'):r['role'] for r in fold_rows if int(r['fold'])==fold}
    assert len(roles)==4000
    tr=[r for r in rows if roles[r['source']]=='train' and decides(r,version)]
    va=[r for r in rows if roles[r['source']]=='val']
    assert all(eligible(r) for r in tr+va), '제외 사진이 분할에 유입됨'
    assert not ({r['source'] for r in tr}&{r['source'] for r in va})
    for cls in ('normal','underfill_50'):
        a=np.array([r['frame_index'] for r in tr if r['label']==cls])
        b=np.array([r['frame_index'] for r in va if r['label']==cls])
        assert len(a)>0 and len(b)>0 and np.min(np.abs(a[:,None]-b[None,:]))>30
    return tr,va


def prepare_cache(args,rows):
    cache=args.cache;cache.mkdir(parents=True,exist_ok=True)
    cfg=dict(manifest=sha(args.bundle/'manifest.csv'),size=args.size,resize='tensor bilinear antialias=True',crop=[250,0,1250,1000],sources=[r['source'] for r in rows],
        area_rule='strictly_greater',minimum=args.rescue_min_area,policy_sha=sha(args.bundle/'area_policy.py'))
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
        assert abs(float((mask>0).mean())-r['mask_area'])<1e-12 and eligible(r,args.rescue_min_area)
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
    config=dict(arch='convnext_tiny',versions=['v1','v2'],version=version,seed=seed,fold=fold,size=args.size,batch=args.batch,rounds=args.rounds,
        train_n=len(train),val_n=len(val),per_round_steps=per_round,total_steps=steps,
        manifest_sha=sha(args.bundle/'manifest.csv'),folds_sha=sha(args.bundle/'folds.csv'),
        trainer_sha=sha(__file__),baseline_sha=sha(args.bundle/'baseline_classifier.py'),
        torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,fast=True,
        lr_backbone=1e-4,lr_head=1e-3,weight_decay=1e-4,augmentation=original.AUGMENT_DESC,
        final_selection='fixed_last_round',rescue_min_area=args.rescue_min_area,area_comparison='>',policy_sha=sha(args.bundle/'area_policy.py'))
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config,'설정/환경 변경: 새 결과 폴더 필요'
    else:save_json(out/'config.json',config)
    if (out/'complete.json').exists():print(f'이미 완료 {version} seed{seed} fold{fold}',flush=True);return
    original.set_seed(seed*100+fold)
    model,head,_=original.build_model('convnext_tiny');model.cuda()
    head_ids={id(p) for p in head.parameters()}
    backbone=[p for p in model.parameters() if id(p) not in head_ids]
    assert not ({id(p) for p in backbone}&head_ids)
    optimizer=torch.optim.AdamW([dict(params=backbone,lr=1e-4),
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
    policy_path=output/'eligibility_policy.json'
    current_min=json.loads(policy_path.read_text())['minimum'] if policy_path.exists() else MIN_AREA
    assert all(float(r['mask_area'])>current_min for r in oof), '제외 사진이 OOF에 유입됨'
    summary=[]
    for version,seed in sorted({(r['version'],r['seed']) for r in oof}):
        predictions=[r for r in oof if r['version']==version and r['seed']==seed]
        assert len(predictions)==len({r['source'] for r in predictions})
        folds=sorted({r['fold'] for r in predictions})
        for cohort in ('all_common','primary','rescue','pipeline_accepted'):
            group=[r for r in predictions if cohort=='all_common' or r['cohort']==cohort or cohort=='pipeline_accepted' and r['pipeline_decides']]
            if not group:continue
            metrics=original.binary_metrics([r['label_id'] for r in group],[r['probability_underfill'] for r in group])
            summary.append(dict(version=version,seed=seed,cohort=cohort,completed_folds=','.join(map(str,folds)),
                                all_folds_complete=folds==[0,1,2,3,4],**metrics))
    write_csv(output/'summary.csv',summary);write_csv(output/'oof_predictions.csv',oof)
    pairs=[]
    for seed in sorted({r['seed'] for r in oof}):
        for a,b in (('v1','v2'),):
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
    for version in ('v1','v2'):
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
        a,av=split(rows,fold_rows,fold,'v1');b,bv=split(rows,fold_rows,fold,'v2')
        assert av==bv and {r['source'] for r in a}<={r['source'] for r in b}

    candidates=read_csv(args.bundle/'manifest.csv')
    excluded=[r for r in candidates if r['original_cohort'] in {'primary','rescue'} and not eligible(r,args.rescue_min_area)]
    assert not ({r['source'] for r in rows}&{r['source'] for r in excluded})
    if args.output is not None:
        args.output.mkdir(parents=True,exist_ok=True)
        write_csv(args.output/'excluded_area.csv',excluded)
        audit=[]
        for version in ('v1','v2'):
            for fold in range(5):
                tr,va=split(rows,fold_rows,fold,version)
                audit.append(dict(version=version,fold=fold,train_n=len(tr),val_n=len(va),
                    train_min_area=min(r['mask_area'] for r in tr),val_min_area=min(r['mask_area'] for r in va),
                    excluded_in_train=0,excluded_in_val=0))
        write_csv(args.output/'eligibility_audit.csv',audit)
        save_json(args.output/'eligibility_policy.json',dict(comparison='>',minimum=args.rescue_min_area,
            eligible_total=len(rows),excluded_area_count=len(excluded),
            excluded_use='no training, validation, feature extraction or prediction'))
    print(f'면적 제외 {len(excluded)}장: 학습/검증/추론 없음. 사용 후보 {len(rows)}장',flush=True)
    if args.verify_only:print('목록/5fold/30frame buffer 검증 완료. 학습 없음.',flush=True);return
    prepare_cache(args,rows)
    if args.prepare_only:print('입력 캐시 완료. 학습 없음.',flush=True);return
    assert torch.cuda.is_available(),'Colab GPU에서 실행해야 함'
    args.output.mkdir(parents=True,exist_ok=True)
    torch.backends.cudnn.benchmark=True
    save_json(args.output/'environment.json',dict(torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,
        gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,fast_nondeterministic=True,main_training='Colab'))
    __import__('shutil').copy2(args.bundle/'preparation.json',args.output/'preparation.json')
    coverage=[]
    for version in ('v1','v2'):
        for cls in ('normal','underfill_50'):
            total=sum(r['label']==cls for r in candidates)
            accepted=sum(r['label']==cls and decides(r,version) for r in rows)
            coverage.append(dict(version=version,label=cls,total=total,accepted=accepted,held=total-accepted,coverage=accepted/total))
    write_csv(args.output/'coverage.csv',coverage)
    save_json(args.output/'run_settings.json',dict(arch='convnext_tiny',versions=['v1','v2'],seeds=args.seeds,folds=args.folds,rescue_min_area=args.rescue_min_area,rounds=args.rounds,size=args.size,batch=args.batch))
    for h in hold_runs(args.bundle,rows,args.output):print(f"보류 {h['version']} {h['label']}: {h['held']}장, 최장 연속 {h['longest_hold_run']}장",flush=True)
    for seed in args.seeds:
        for fold in args.folds:
            for version in ('v1','v2'):
                run(args,rows,fold_rows,version,seed,fold);aggregate(args.output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--originals',type=Path)
    p.add_argument('--cache',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--size',type=int,default=512);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--rounds',type=int,default=20);p.add_argument('--seeds',type=int,nargs='+',default=[0])
    p.add_argument('--folds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--rescue-min-area',type=float,default=MIN_AREA,help='사용 면적은 이 값 초과. 기본 0.1, 10% 이하 사용 불가')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.batch<2 or args.size<32 or args.rounds<1:p.error('batch>=2, size>=32, rounds>=1')
    if not args.verify_only and (args.originals is None or args.cache is None):p.error('--originals --cache 필요')
    if not (args.verify_only or args.prepare_only) and args.output is None:p.error('--output 필요')
    if not MIN_AREA<=args.rescue_min_area<1:p.error('면적 하한은 0.1 이상 1 미만')
    main(args)
