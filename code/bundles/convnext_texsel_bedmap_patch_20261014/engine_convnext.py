"""공통 ConvNeXt 학습 엔진. 실행은 train_convnext_texsel_bedmap.py를 사용한다."""

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
    config=dict(arch='convnext_tiny',versions=['v6_tex_bedmap'],adapter_sha=sha(Path(__file__).with_name('train_convnext_texsel_bedmap.py')),texture_manifest_sha=sha(args.texture/'texture_manifest.csv'),version=version,seed=seed,fold=fold,size=args.size,batch=args.batch,rounds=args.rounds,
        train_n=len(train),val_n=len(val),per_round_steps=per_round,total_steps=steps,
        manifest_sha=sha(args.bundle/'manifest.csv'),folds_sha=sha(args.bundle/'folds.csv'),
        trainer_sha=sha(__file__),baseline_sha=sha(args.bundle/'baseline_classifier.py'),
        torch=str(torch.__version__),torchvision=__import__('torchvision').__version__,fast=True,
        lr_backbone=1e-4,lr_head=1e-3,weight_decay=1e-4,augmentation=original.AUGMENT_DESC,
        final_selection='fixed_last_round',rescue_min_area=args.rescue_min_area,area_comparison='>',policy_sha=sha(args.bundle/'area_policy.py'))
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config,'설정/환경 변경: 새 결과 폴더 필요'
    else:save_json(out/'config.json',config)
    if (out/'complete.json').exists():
        assert (out/'model_state.pt').exists() and (out/'predictions.csv').exists(), '완료 결과 파일이 누락됨'
        print(f'이미 완료 {version} seed{seed} fold{fold}',flush=True);return
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
