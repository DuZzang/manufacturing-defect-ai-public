"""언더필 표면 50장으로 256px RGB DCGAN을 시험하는 예비 실험.

입력 폴더는 평가 사진을 제외한 개발용 학습 사진 50장만 포함해야 한다.
이 코드는 사진이 실제 언더필을 잘 보여 주는지 자동으로 보증하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image


class Generator(nn.Module):
    """난수 128개를 256×256 RGB 이미지로 바꾼다."""

    def __init__(self, base: int = 32, latent_dim: int = 128):
        super().__init__()
        channels = [base * 16, base * 8, base * 4, base * 2, base, base // 2]
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(latent_dim, channels[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels[0]), nn.ReLU(True),
        ]
        for prev, nxt in zip(channels, channels[1:]):
            layers.extend([nn.ConvTranspose2d(prev, nxt, 4, 2, 1, bias=False),
                           nn.BatchNorm2d(nxt), nn.ReLU(True)])
        layers.extend([nn.ConvTranspose2d(channels[-1], 3, 4, 2, 1, bias=False), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """실제/생성 이미지를 구분할 학습 점수(logit)를 만든다."""

    def __init__(self, base: int = 32):
        super().__init__()
        channels = [base // 2, base, base * 2, base * 4, base * 8, base * 16]
        layers: list[nn.Module] = [nn.Conv2d(3, channels[0], 4, 2, 1, bias=False),
                                   nn.LeakyReLU(0.2, inplace=True)]
        for prev, nxt in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(prev, nxt, 4, 2, 1, bias=False),
                           nn.BatchNorm2d(nxt), nn.LeakyReLU(0.2, inplace=True)])
        layers.append(nn.Conv2d(channels[-1], 1, 4, 1, 0, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).reshape(-1)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_images(folder: Path) -> tuple[torch.Tensor, list[dict], str]:
    if not folder.is_dir():
        raise ValueError(f'학습 사진 폴더를 찾을 수 없습니다: {folder}')
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if len(paths) != 50:
        raise ValueError(f'학습 이미지가 정확히 50장이어야 합니다. 현재 {len(paths)}장: {folder}')
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tensors, manifest = [], []
    for path in paths:
        try:
            with Image.open(path) as original:
                img = ImageOps.exif_transpose(original).convert('RGB')
                if img.width != img.height:
                    raise ValueError(f'정사각형이 아닙니다 ({img.width}x{img.height})')
                tensors.append(transform(img))
                manifest.append({'filename': path.name, 'path': str(path.resolve()),
                                 'width': img.width, 'height': img.height, 'sha256': hash_file(path)})
        except Exception as exc:
            raise ValueError(f'학습 사진을 읽을 수 없습니다: {path}: {exc}') from exc
    # 이름과 픽셀 파일의 지문으로, 재개 시 사진 구성 변경을 감지한다.
    signature = hashlib.sha256(json.dumps(
        [(row['filename'], row['sha256']) for row in manifest],
        ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    return torch.stack(tensors), manifest, signature


def select_device(choice: str) -> torch.device:
    if choice == 'auto':
        choice = 'cuda' if torch.cuda.is_available() else 'cpu'
    if choice == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU를 사용할 수 없습니다. Colab GPU 연결을 확인하거나 --device cpu를 사용하세요.')
    if choice == 'cpu':
        print('주의: CPU 학습은 오래 걸릴 수 있습니다.', flush=True)
    return torch.device(choice)


def train(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.batch < 2 or args.num_generate < 1 or args.save_every < 1:
        raise ValueError('--steps/--num-generate/--save-every는 1 이상, --batch는 2 이상이어야 합니다.')
    images, manifest, dataset_sig = load_images(Path(args.data))
    device = select_device(args.device)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / 'checkpoint.pt'
    results = out / 'generated'
    config = {'schema_version': 1, 'architecture': 'dcgan_rgb256', 'image_size': 256,
              'latent_dim': 128, 'base': args.base, 'batch': args.batch, 'seed': args.seed,
              'learning_rate': 0.0002, 'betas': [0.5, 0.999],
              'dataset_fingerprint': dataset_sig, 'training_image_count': len(manifest),
              'preprocessing': 'RGB, EXIF 반영, 정사각 입력을 bilinear 256으로 조정, [-1,1] 정규화',
              'augmentation': '없음', 'evaluation_warning': '예비 생성 실험. 품질과 유용성은 별도 확인 필요.'}
    if args.verify_only:
        print(json.dumps({'status': 'verified_no_training', 'device': str(device),
                          'input_count': len(manifest), 'dataset_fingerprint': dataset_sig,
                          'image_shape': list(images.shape)}, ensure_ascii=False, indent=2), flush=True)
        return

    existing_pngs = list(results.glob('*.png')) if results.exists() else []
    if existing_pngs and not checkpoints.exists():
        raise RuntimeError('체크포인트 없는 생성 PNG가 이미 있습니다. 다른 --out 폴더를 지정하세요.')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    generator = Generator(args.base).to(device)
    discriminator = Discriminator(args.base).to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    sample_rng = torch.Generator(device='cpu').manual_seed(args.seed + 211)
    preview_rng = torch.Generator(device='cpu').manual_seed(args.seed + 337)
    fixed_z = torch.randn(16, 128, 1, 1, generator=preview_rng)
    history: list[dict] = []
    start = 0
    if checkpoints.exists():
        state = torch.load(checkpoints, map_location='cpu', weights_only=True)
        if state.get('config') != config:
            raise RuntimeError('기존 체크포인트의 사진·시드·배치·설정이 다릅니다. 새 --out 폴더를 사용하세요.')
        start = int(state['step'])
        if args.steps < start:
            raise RuntimeError(f'요청 단계({args.steps})가 기존 체크포인트({start})보다 작습니다.')
        if existing_pngs and args.steps > start:
            raise RuntimeError('생성 결과가 이미 있으므로 기존 폴더에서 학습을 연장하지 않습니다. 새 --out 폴더를 사용하세요.')
        generator.load_state_dict(state['generator'])
        discriminator.load_state_dict(state['discriminator'])
        opt_g.load_state_dict(state['optimizer_g'])
        opt_d.load_state_dict(state['optimizer_d'])
        sample_rng.set_state(state['sample_rng'])
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        elif state.get('device') != str(device):
            print('주의: 실행 장치가 변경되어 중단 전과 완전히 같은 난수 재현을 보장하지 않습니다.', flush=True)
        fixed_z = state['fixed_z']
        history = state.get('history', [])
        print(f'체크포인트에서 재개: {start}/{args.steps} step', flush=True)
    elif any(out.iterdir()):
        raise RuntimeError(f'체크포인트가 없는 기존 출력 폴더입니다. 새 --out 폴더를 사용하세요: {out}')

    save_json(out / 'training_manifest.json', {'data_folder': str(Path(args.data).resolve()),
                                                'fingerprint': dataset_sig, 'images': manifest})
    save_json(out / 'gan_config.json', config)
    preview_dir = out / 'previews'
    preview_dir.mkdir(exist_ok=True)

    def checkpoint(step: int) -> None:
        payload = {'config': config, 'step': step, 'device': str(device),
                   'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(),
                   'optimizer_g': opt_g.state_dict(), 'optimizer_d': opt_d.state_dict(),
                   'sample_rng': sample_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                   'fixed_z': fixed_z, 'history': history}
        tmp = out / 'checkpoint.pt.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, checkpoints)

    step = start
    try:
        save_json(out / 'progress.json', {'status': 'running', 'step': start, 'target_steps': args.steps,
                                          'history': history})
        while step < args.steps:
            generator.train()
            discriminator.train()
            idx = torch.randint(len(images), (args.batch,), generator=sample_rng)
            real = images[idx].to(device)
            z = torch.randn(args.batch, 128, 1, 1, device=device)
            fake = generator(z)

            # 판별기는 실제에는 1, 생성에는 0에 가까운 점수를 내도록 학습한다.
            for param in discriminator.parameters():
                param.requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)
            d_real = discriminator(real)
            d_fake = discriminator(fake.detach())
            loss_d = (loss_fn(d_real, torch.ones_like(d_real)) +
                      loss_fn(d_fake, torch.zeros_like(d_fake))) / 2.0
            loss_d.backward()
            opt_d.step()

            # 생성기는 판별기가 생성 이미지를 실제에 가깝다고 보게 학습한다.
            for param in discriminator.parameters():
                param.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            g_pred = discriminator(fake)
            loss_g = loss_fn(g_pred, torch.ones_like(g_pred))
            loss_g.backward()
            opt_g.step()
            step += 1
            if not torch.isfinite(loss_d).item() or not torch.isfinite(loss_g).item():
                raise RuntimeError(f'{step} step에서 손실이 유한하지 않습니다. 학습을 중단합니다.')

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                record = {'step': step, 'loss_d': round(loss_d.item(), 6), 'loss_g': round(loss_g.item(), 6)}
                history.append(record)
                print(f"step {step}/{args.steps} | D={record['loss_d']:.4f} G={record['loss_g']:.4f}", flush=True)
                save_json(out / 'progress.json', {'status': 'running', 'step': step,
                                                  'target_steps': args.steps, 'history': history})
            if step % args.save_every == 0 or step == args.steps:
                checkpoint(step)
                generator.eval()
                with torch.no_grad():
                    previews = generator(fixed_z.to(device)).cpu()
                save_image(previews, preview_dir / f'step_{step:06d}.png',
                           nrow=4, normalize=True, value_range=(-1, 1))

        results.mkdir(exist_ok=True)
        info_path = out / 'generation_info.json'
        generation_info = {'seed': args.seed + 50000, 'checkpoint_step': step,
                           'num_generate': args.num_generate, 'dataset_fingerprint': dataset_sig}
        if info_path.exists():
            old = json.loads(info_path.read_text(encoding='utf-8'))
            if old != generation_info:
                raise RuntimeError('기존 생성 결과와 조건이 다릅니다. 새 --out 폴더를 사용하세요.')
        elif existing_pngs:
            raise RuntimeError('생성 기록이 없는 PNG가 이미 있습니다. 새 --out 폴더를 사용하세요.')
        else:
            save_json(info_path, generation_info)

        final_rng = torch.Generator(device='cpu').manual_seed(args.seed + 50000)
        generator.eval()
        with torch.no_grad():
            for i in range(args.num_generate):
                z = torch.randn(1, 128, 1, 1, generator=final_rng).to(device)
                dest = results / f'generated_{i + 1:03d}.png'
                if not dest.exists():
                    save_image(generator(z).cpu(), dest, normalize=True, value_range=(-1, 1))
        save_json(out / 'progress.json', {'status': 'completed', 'step': step,
                                          'target_steps': args.steps, 'generated': args.num_generate,
                                          'generated_dir': str(results), 'history': history})
        print(f'완료: {args.num_generate}장 저장 -> {results}', flush=True)
    except KeyboardInterrupt:
        save_json(out / 'progress.json', {'status': 'interrupted', 'step': step,
                                          'last_checkpoint': str(checkpoints) if checkpoints.exists() else None,
                                          'history': history})
        raise
    except Exception as exc:
        save_json(out / 'progress.json', {'status': 'failed', 'step': step, 'error': str(exc),
                                          'last_checkpoint': str(checkpoints) if checkpoints.exists() else None,
                                          'history': history})
        raise


def smoke_test() -> None:
    """임시 합성 사진만 이용하는 CPU 1-step 파이프라인 점검."""
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with tempfile.TemporaryDirectory(prefix='gan_smoke_', dir=Path(__file__).resolve().parent) as temp:
        root = Path(temp)
        inputs = root / 'inputs'
        inputs.mkdir()
        for i in range(50):
            Image.new('RGB', (32, 32), (i * 5 % 256, i * 3 % 256, i * 7 % 256)).save(inputs / f'{i:03d}.png')
        args = argparse.Namespace(data=str(inputs), out=str(root / 'out'), steps=1, seed=123,
                                  num_generate=2, device='cpu', batch=2, save_every=1,
                                  log_every=1, base=8, verify_only=False)
        train(args)
        generated = list((root / 'out' / 'generated').glob('*.png'))
        assert len(generated) == 2
        with Image.open(generated[0]) as sample:
            assert sample.size == (256, 256) and sample.mode == 'RGB'
        # 이미 완료된 같은 조건을 재실행해 체크포인트 복원 경로도 점검한다.
        train(args)
        print('SMOKE_TEST_OK: 합성 50장, CPU 1-step, 256px 생성, 체크포인트 재개', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='실제 언더필 학습 사진 50장 -> DCGAN 이미지 생성 (예비 실험)')
    parser.add_argument('--data', help='개발용 학습 사진 50장 폴더')
    parser.add_argument('--out', help='새 출력 폴더. 동일 설정·동일 사진이면 체크포인트에서 재개')
    parser.add_argument('--steps', type=int, default=1500, help='총 업데이트 수 (1500은 검증되지 않은 예비값)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-generate', type=int, default=50)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--base', type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument('--verify-only', action='store_true', help='사진·설정만 확인하고 학습하지 않음')
    parser.add_argument('--smoke-test', action='store_true', help='임시 합성 사진으로 CPU 1-step 자체 점검')
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if not args.data or not args.out:
        parser.error('--data와 --out을 지정해야 합니다.')
    if args.log_every < 1 or args.base < 8 or args.base % 2:
        parser.error('--log-every는 1 이상, --base는 8 이상의 짝수여야 합니다.')
    train(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'오류: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
