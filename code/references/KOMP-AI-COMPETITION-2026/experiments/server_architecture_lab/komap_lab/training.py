import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import FullResolutionDataset, NativePatchDataset, manifest
from .evaluation import evaluate
from .loss import R022Loss
from .models import SegmentationModel, load_encoder
from .runtime import (capture_rng, cpu_tree, environment, restore_rng, resume_guard, run_directory,
                      save_checkpoint, seed_everything, write_json)
from .ssl import MultiStageSSL, TrainViews, nt_xent


def checkpoint_base(config, data_manifest, device, phase, initialization):
    return {'config': config, 'data_manifest': data_manifest, 'environment': environment(device),
            'phase': phase, 'initialization': initialization}


def save_history(output, history):
    write_json(output / 'history.json', history)
    if history:
        temp = output / 'history.csv.tmp'
        with temp.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        temp.replace(output / 'history.csv')


def train(config, data, output, device, resume=False, encoder_weights=None, encoder_source=None):
    if config['model']['mamba_blocks'] and device.type != 'cuda':
        raise ValueError('C10 requires CUDA')
    seed_everything(config['train']['seed'], config['train']['deterministic'])
    with run_directory(output, resume) as output:
        t = config['train']
        data_manifest = manifest(data, patch=t['target'])
        previous = torch.load(output / 'last.pt', map_location='cpu', weights_only=True) if resume else None
        if previous:
            resume_guard(previous, config, data_manifest, 'segmentation', device)
        initialization = previous['initialization'] if previous else {'kind': config['model']['initialization']}
        kind = config['model']['initialization']
        if not resume and kind in ('micronet', 'ssl') and encoder_weights is None:
            raise ValueError(f'{kind} initialization needs --encoder-weights; no ImageNet fallback')
        if not resume and kind == 'micronet' and not encoder_source:
            raise ValueError('MicroNet needs --encoder-source identifying version/training variant')
        if not resume and kind not in ('micronet', 'ssl') and encoder_weights is not None:
            raise ValueError('--encoder-weights only valid for micronet/ssl configurations')
        if resume and encoder_weights is not None:
            raise ValueError('Resume uses stored weights; omit --encoder-weights')
        model = SegmentationModel(config, pretrained=not resume and kind == 'imagenet')
        if not resume and kind in ('micronet', 'ssl'):
            initialization = load_encoder(model.encoder, encoder_weights, kind)
            initialization['source'] = encoder_source
            if kind == 'ssl':
                train_images = [{k: v for k, v in row.items() if k != 'mask_sha256'} for row in data_manifest['train']]
                if initialization['metadata'].get('train_images') != train_images:
                    raise ValueError('SSL Train images differ from supervised Train; transfer rejected')
        model.to(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        criterion = R022Loss(**config['loss']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t['epochs'])
        dataset = NativePatchDataset(data, context=t['patch'], target=t['target'], stride=t['stride'],
                                     rare_fraction=t['rare_fraction'], enriched_probability=t['enriched_probability'],
                                     rare_classes=t['rare_classes'], samples_per_epoch=t['samples_per_epoch'])
        valid = FullResolutionDataset(data, 'valid')
        generator = torch.Generator().manual_seed(t['seed'])
        loader = DataLoader(dataset, batch_size=t['batch_size'], shuffle=True, num_workers=0, generator=generator,
                            pin_memory=device.type == 'cuda')
        history, best, best_epoch, start_epoch, best_model = [], -1., 0, 1, None
        if previous:
            model.load_state_dict(previous['model'], strict=True)
            optimizer.load_state_dict(previous['optimizer'])
            scheduler.load_state_dict(previous['scheduler'])
            history, best, best_epoch = previous['history'], previous['best_miou'], previous['best_epoch']
            best_model = previous['best_model']
            start_epoch = previous['epoch'] + 1
            restore_rng(previous['rng'], generator)
            # last.pt is the authoritative epoch transaction, including best weights.
            save_checkpoint(output / 'best.pt', {**checkpoint_base(config, data_manifest, device, 'segmentation', initialization),
                                                  'epoch': best_epoch, 'model': best_model,
                                                  'best_miou': best, 'best_epoch': best_epoch})
        base = checkpoint_base(config, data_manifest, device, 'segmentation', initialization)
        write_json(output / 'config.json', config)
        write_json(output / 'data_manifest.json', data_manifest)
        write_json(output / 'environment.json', base['environment'])
        print(json.dumps({'id': config['id'], 'device': str(device), 'parameters': sum(p.numel() for p in model.parameters()),
                          'start_epoch': start_epoch, 'rich_sources': len(dataset.enriched),
                          'rich_tiles': sum(map(len, dataset.enriched.values()))}), flush=True)
        try:
            for epoch in range(start_epoch, t['epochs'] + 1):
                started, loss_sum, examples = time.perf_counter(), 0., 0
                model.train()
                for step, (images, targets, _) in enumerate(loader, 1):
                    images, targets = images.to(device), targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(images), targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at epoch {epoch} step {step}')
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(images)
                    examples += len(images)
                    if step == 1 or step % 10 == 0 or step == len(loader):
                        write_json(output / 'status.json', {'state': 'training', 'epoch': epoch, 'epochs': t['epochs'],
                                  'step': step, 'steps': len(loader), 'train_loss': loss_sum / examples,
                                  'best_miou': best if best >= 0 else None, 'best_epoch': best_epoch})
                validation = evaluate(model, valid, config, device)
                score = validation['miou']
                if not math.isfinite(score):
                    raise FloatingPointError('Nonfinite validation score')
                record = {'epoch': epoch, 'train_loss': loss_sum / examples, 'valid_single_miou': score,
                          'primary_si_iou': validation['class_iou'][0], 'al3ni_iou': validation['class_iou'][1],
                          'eutectic_si_iou': validation['class_iou'][2], 'al_iou': validation['class_iou'][3],
                          'lr': optimizer.param_groups[0]['lr'], 'epoch_seconds': time.perf_counter() - started}
                history.append(record)
                scheduler.step()
                if score > best:
                    best, best_epoch = score, epoch
                    # Clone: cpu_tree alone aliases CPU tensors during CPU training.
                    best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                state = {**base, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'rng': capture_rng(generator), 'history': history,
                         'best_miou': best, 'best_epoch': best_epoch, 'best_model': best_model}
                save_checkpoint(output / 'last.pt', state)
                save_checkpoint(output / 'best.pt', {**base, 'epoch': best_epoch, 'model': best_model,
                                                    'best_miou': best, 'best_epoch': best_epoch})
                save_history(output, history)
                print(json.dumps({**record, 'best_miou': best, 'best_epoch': best_epoch}), flush=True)
            best_checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(best_checkpoint['model'], strict=True)
            final = {'id': config['id'], 'best_epoch': best_epoch, 'initialization': initialization}
            for d4 in (False, True) if config['evaluation']['final_d4'] else (False,):
                name = 'd4' if d4 else 'single'
                result = evaluate(model, valid, config, device, d4=d4, export=output / 'valid_masks' / name, diagnostic=True)
                write_json(output / f'valid_{name}.json', result)
                final[name] = result
            final['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
            write_json(output / 'summary.json', final)
            write_json(output / 'status.json', {'state': 'completed', 'epochs': t['epochs'], 'best_epoch': best_epoch,
                                               'best_single_miou': best, 'final_d4_miou': final.get('d4', {}).get('miou')})
        except (Exception, KeyboardInterrupt) as exc:
            write_json(output / 'status.json', {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                               'error': str(exc), 'resume_from_last_completed_epoch': True})
            raise


def pretrain_ssl(config, data, output, device, resume=False):
    if config['model']['initialization'] not in ('ssl', 'imagenet', 'none'):
        raise ValueError('SSL pretraining supports ImageNet start or random start, not MicroNet')
    seed_everything(config['train']['seed'], config['train']['deterministic'])
    with run_directory(output, resume) as output:
        t, s = config['train'], config['ssl']
        data_manifest = manifest(data, include_valid=False, patch=s['patch'])
        views = TrainViews(data, s['patch'])
        if s['batch_size'] > len(views.images):
            raise ValueError('SSL batch_size exceeds distinct Train source count')
        previous = torch.load(output / 'last.pt', map_location='cpu', weights_only=True) if resume else None
        if previous:
            resume_guard(previous, config, data_manifest, 'ssl', device)
        imagenet = config['model']['initialization'] != 'none'
        model = MultiStageSSL(config, pretrained=not resume and imagenet).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=s['lr'], weight_decay=s['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s['epochs'])
        history, start_epoch = [], 1
        if previous:
            model.load_state_dict(previous['model'], strict=True)
            optimizer.load_state_dict(previous['optimizer'])
            scheduler.load_state_dict(previous['scheduler'])
            history, start_epoch = previous['history'], previous['epoch'] + 1
            restore_rng(previous['rng'])
        base = checkpoint_base(config, data_manifest, device, 'ssl', {'kind': 'imagenet' if imagenet else 'none'})
        write_json(output / 'config.json', config)
        write_json(output / 'data_manifest.json', data_manifest)
        write_json(output / 'environment.json', base['environment'])
        try:
            for epoch in range(start_epoch, s['epochs'] + 1):
                started, loss_sum = time.perf_counter(), 0.
                model.train()
                for step in range(1, s['steps_per_epoch'] + 1):
                    first, second = views.batch(s['batch_size'])
                    optimizer.zero_grad(set_to_none=True)
                    # Both views share one BN pass and an equal-sized contrastive batch.
                    embeddings = model(torch.cat([first, second]).to(device))
                    loss = nt_xent(*embeddings.chunk(2), temperature=s['temperature'])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite SSL loss')
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach())
                    if step == 1 or step % 10 == 0 or step == s['steps_per_epoch']:
                        write_json(output / 'status.json', {'state': 'ssl_training', 'epoch': epoch, 'epochs': s['epochs'],
                                                           'step': step, 'steps': s['steps_per_epoch'], 'loss': loss_sum / step})
                history.append({'epoch': epoch, 'ssl_loss': loss_sum / s['steps_per_epoch'],
                                'lr': optimizer.param_groups[0]['lr'], 'epoch_seconds': time.perf_counter() - started})
                scheduler.step()
                save_checkpoint(output / 'last.pt', {**base, 'epoch': epoch, 'model': model.state_dict(),
                                                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                                                     'history': history, 'rng': capture_rng()})
                save_history(output, history)
                print(json.dumps(history[-1]), flush=True)
            # No validation-based selection: fixed final SSL epoch, Train images only.
            save_checkpoint(output / 'encoder.pt', {'encoder': model.encoder.state_dict(),
                             'metadata': {'pretraining': 'multi-stage-ssl', 'backbone': config['model']['backbone'],
                                          'epochs': s['epochs'], 'train_images': data_manifest['train'],
                                          'normalization': config['normalization'], 'config': config,
                                          'environment': base['environment']}})
            write_json(output / 'status.json', {'state': 'completed', 'epochs': s['epochs'], 'encoder': 'encoder.pt'})
        except (Exception, KeyboardInterrupt) as exc:
            write_json(output / 'status.json', {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                               'error': str(exc), 'resume_from_last_completed_epoch': True})
            raise
