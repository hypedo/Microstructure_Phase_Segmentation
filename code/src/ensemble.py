"""Native-resolution checkpoint comparison, cached probabilities, and local exports.

Run from komap, for example:
  python src/ensemble.py --data data --checkpoints runs/a/best.pt runs/b/best.pt \
      --split valid --out runs/ensemble --tta
  python src/ensemble.py --data data --checkpoints runs/a/best.pt runs/b/best.pt \
      --split test --out artifacts/test_ensemble --weights 0.5 0.5 --tta

The optional development-set search chooses among a small, fixed set of simplex
weights. Its leave-one-image-out result assesses weight selection only; it does
not undo earlier checkpoint selection or training on those images. Nothing is
uploaded or submitted. Torch and train are imported only by main, so reporting
and cache checks can also run in a lightweight NumPy environment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import warnings
import zipfile

import numpy as np

from metrics import CLASS_NAMES, PALETTE, confusion, encode_mask, score_confusion, summarize

SCHEMA_VERSION = 1
METRIC_KEYS = ('miou_absent_zero', 'miou_absent_one', 'miou_present_only')


def clean_json(value):
    """Emit valid JSON even when a class is absent from every evaluated image."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize_weights(weights, count):
    if count < 1:
        raise ValueError('At least one model is required')
    result = np.ones(count, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError('Provide exactly one finite weight per checkpoint, in checkpoint order')
    if np.any(result < 0) or result.sum() <= 0:
        raise ValueError('Weights must be nonnegative and have a positive sum')
    return result / result.sum()


def validate_probabilities(probabilities, shape):
    p = np.asarray(probabilities)
    expected = (len(CLASS_NAMES), *tuple(shape))
    if p.shape != expected:
        raise ValueError(f'Probability shape {p.shape} != expected {expected}')
    if not np.isfinite(p).all() or p.min() < -1e-5 or p.max() > 1.001:
        raise ValueError('Probabilities contain invalid values')
    p = p.astype(np.float32)
    totals = p.sum(axis=0)
    if not np.allclose(totals, 1.0, rtol=0, atol=0.005):
        raise ValueError('Class probabilities must sum to one at every pixel')
    # FP16 storage introduces small rounding differences; normalize identically
    # for single-model reporting and for ensembles.
    p /= totals[None]
    return p


def write_probability_cache(path, probabilities, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = validate_probabilities(probabilities, metadata['shape']).astype(np.float16)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, probabilities=p,
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(path)


def read_probability_cache(path, expected_metadata):
    with np.load(path, allow_pickle=False) as archive:
        stored = json.loads(str(archive['metadata'].item()))
        if stored != expected_metadata:
            raise ValueError(f'Cache provenance mismatch: {path}')
        p = archive['probabilities']
        if p.dtype != np.float16:
            raise ValueError(f'Cache probabilities must be FP16: {path}')
        return validate_probabilities(p, expected_metadata['shape'])


def weighted_probabilities(probabilities, weights):
    if not probabilities:
        raise ValueError('No probability arrays supplied')
    weights = normalize_weights(weights, len(probabilities))
    shape = probabilities[0].shape
    output = np.zeros(shape, dtype=np.float32)
    for p, weight in zip(probabilities, weights):
        if p.shape != shape:
            raise ValueError('All model probabilities must have identical native-resolution shape')
        if weight:
            output += p * np.float32(weight)
    return output


def bootstrap_mean(values, repetitions=10000, seed=2026):
    """Percentile interval from resampling whole images with replacement."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Bootstrap requires a nonempty finite image-score vector')
    if repetitions < 100:
        raise ValueError('At least 100 bootstrap repetitions are required')
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    # Keep memory bounded if callers later evaluate a larger image collection.
    for start in range(0, repetitions, 1000):
        size = min(1000, repetitions - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {'mean': float(values.mean()), 'lower': float(low), 'upper': float(high),
            'confidence': .95, 'repetitions': repetitions, 'seed': seed,
            'resampling_unit': 'original_image', 'n_images': len(values)}


def summarize_with_intervals(rows, repetitions=10000, seed=2026):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # Undefined all-absent class means -> JSON null.
        result = summarize(rows)
    result['image_bootstrap_95ci'] = {
        key: bootstrap_mean([row[key] for row in rows], repetitions, seed)
        for key in METRIC_KEYS
    }
    result['interval_scope'] = (
        'Conditional on fixed checkpoints and inference settings. Image bootstrap does not '
        'account for earlier model selection, related specimens, or development-weight fitting.'
    )
    return clean_json(result)


def paired_differences(reference_rows, comparison_rows, repetitions=10000, seed=2026):
    if [r['id'] for r in reference_rows] != [r['id'] for r in comparison_rows]:
        raise ValueError('Paired comparisons require the same image IDs in the same order')
    return {key: bootstrap_mean([b[key] - a[key] for a, b in zip(reference_rows, comparison_rows)],
                                repetitions, seed) for key in METRIC_KEYS}


def modest_simplex_candidates(count):
    """A prespecified grid near equal weights, rather than an unconstrained search."""
    equal = normalize_weights(None, count)
    candidates = [equal]
    if count > 1:
        for strength in (.15, .30):
            for model_index in range(count):
                candidate = equal * (1 - strength)
                candidate[model_index] += strength
                candidates.append(candidate)
    return np.stack(candidates)


def fit_candidate_weights(scores, candidates, ids, metric_index=0):
    """Choose fixed candidate weights and audit weight choice by leave-one-image-out.

    scores[image, candidate, convention] is calculated from full-resolution masks.
    The held-out image never participates in the weight-selection mean for its fold.
    """
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    if scores.shape != (len(ids), len(candidates), len(METRIC_KEYS)):
        raise ValueError('Unexpected candidate-score table shape')
    if len(ids) < 2 or len(set(ids)) != len(ids) or not np.isfinite(scores).all():
        raise ValueError('Weight selection requires distinct, finite development-image scores')
    objective = scores[:, :, metric_index]
    best = int(np.argmax(objective.mean(axis=0)))  # Equal weights win exact ties.
    total = objective.sum(axis=0)
    loo_scores, folds = [], []
    for index, iid in enumerate(ids):
        selected = int(np.argmax((total - objective[index]) / (len(ids) - 1)))
        fold_values = scores[index, selected]
        loo_scores.append(fold_values)
        folds.append({'held_out_id': iid, 'candidate_index': selected,
                      'weights': candidates[selected].tolist(),
                      **{key: float(fold_values[j]) for j, key in enumerate(METRIC_KEYS)}})
    loo_scores = np.asarray(loo_scores)
    return {
        'selected_index': best, 'weights': candidates[best].tolist(),
        'development_ids': list(ids), 'selection_metric': METRIC_KEYS[metric_index],
        'candidate_weights': candidates.tolist(),
        'candidate_development_means': scores.mean(axis=0).tolist(),
        'candidate_score_convention_order': list(METRIC_KEYS),
        'leave_one_image_out': {
            **{key: float(loo_scores[:, j].mean()) for j, key in enumerate(METRIC_KEYS)},
            'per_image': folds,
            'scope': 'Weight-selection audit only; all folds reuse the same trained checkpoints.',
        },
        'caution': (
            'The fitted weights use development labels. Full-development scores are in-sample. '
            'LOO does not remove leakage from earlier checkpoint selection on those images. '
            'Scores outside these IDs are held out from weight fitting only, not necessarily '
            'from previous architecture or checkpoint decisions.'
        ),
    }


def sample_records(data, split, image_id):
    from PIL import Image
    records = []
    ids = set()
    for path in sorted((Path(data) / split / 'images').glob('*_image.png')):
        iid = image_id(path)
        if iid in ids:
            raise ValueError(f'Duplicate image ID: {iid}')
        ids.add(iid)
        with Image.open(path) as im:
            shape = [im.height, im.width]
        mask = Path(data) / split / 'masks' / f'{iid}_mask.png'
        if split == 'valid' and not mask.exists():
            raise FileNotFoundError(mask)
        records.append({'id': iid, 'path': path, 'shape': shape,
                        'image_sha256': file_sha256(path),
                        'mask': mask if split == 'valid' else None})
    if not records:
        raise ValueError(f'No *_image.png files found in {data}/{split}/images')
    return records


def score_prediction(sample, prediction, read_image):
    target = encode_mask(read_image(sample['mask']))
    row = score_confusion(confusion(target, prediction))
    row['id'] = sample['id']
    return row


def export_predictions(destination, sample, prediction):
    from PIL import Image
    destination = Path(destination)
    rgb_path = destination / 'rgb' / f"{sample['id']}_mask.png"
    index_path = destination / 'indices' / f"{sample['id']}_mask.png"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if prediction.shape != tuple(sample['shape']):
        raise ValueError('Refusing to export a mask that differs from native image dimensions')
    Image.fromarray(PALETTE[prediction]).save(rgb_path)
    Image.fromarray(prediction).save(index_path)
    return {'id': sample['id'], 'input': str(sample['path'].resolve()),
            'shape': sample['shape'], 'rgb_mask': str(rgb_path.resolve()),
            'class_index_mask': str(index_path.resolve())}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', required=True, nargs='+')
    parser.add_argument('--data', default='data')
    parser.add_argument('--split', choices=['valid', 'test'], default='valid')
    parser.add_argument('--out', required=True)
    parser.add_argument('--weights', type=float, nargs='+', help='Nonnegative weights in checkpoint order; default equal')
    parser.add_argument('--tta', action='store_true', help='Eight D4 rotations/reflections via train.tiled_probabilities')
    parser.add_argument('--tile', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='auto', help='auto, cpu, mps, cuda, or cuda:N')
    parser.add_argument('--cache-dir', help='Shared cache root; defaults to OUT/cache')
    parser.add_argument('--overwrite-cache', action='store_true')
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--optimize-weights', action='store_true', help='Optional small prespecified simplex search on explicit development IDs')
    parser.add_argument('--development-ids', help='JSON list of >=4 validation-image IDs used only for weight fitting')
    parser.add_argument('--selection-metric', choices=METRIC_KEYS, default=METRIC_KEYS[0])
    args = parser.parse_args()
    if args.tile < 64 or args.tile % 32 or not (0 <= args.overlap < args.tile) or args.batch < 1:
        parser.error('tile must be >=64 and divisible by 32; 0 <= overlap < tile; batch >=1')
    if args.bootstrap_samples < 100:
        parser.error('--bootstrap-samples must be >=100')
    if args.optimize_weights and (args.split != 'valid' or not args.development_ids or args.weights is not None):
        parser.error('--optimize-weights requires --split valid and --development-ids, and excludes --weights')
    if args.development_ids and not args.optimize_weights:
        parser.error('--development-ids is used only with --optimize-weights')
    try:
        normalize_weights(args.weights, len(args.checkpoints))
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    import torch
    import segmentation_models_pytorch as smp
    from train import build_model, image_id, read_image, tiled_probabilities

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir) if args.cache_dir else out / 'cache'
    automatic = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(automatic if args.device == 'auto' else args.device)
    torch.set_num_threads(4)
    samples = sample_records(args.data, args.split, image_id)
    development_ids = None
    if args.optimize_weights:
        development_ids = json.loads(Path(args.development_ids).read_text())
        if (not isinstance(development_ids, list) or len(development_ids) < 4
                or any(not isinstance(iid, str) for iid in development_ids)
                or len(set(development_ids)) != len(development_ids)):
            raise ValueError('Development IDs must be a JSON list of >=4 distinct strings')
        unknown = set(development_ids) - {sample['id'] for sample in samples}
        if unknown:
            raise ValueError(f'Unknown development IDs: {sorted(unknown)}')

    report = {
        'schema_version': SCHEMA_VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
        'split': args.split, 'n_images': len(samples), 'arguments': vars(args),
        'metric_definition': 'Mean over original images of mean over four phase IoUs; pooled IoU is diagnostic only.',
        'absent_class_note': 'Official public instructions do not define union=0. All three conventions are reported.',
        'class_names': CLASS_NAMES, 'palette': PALETTE.tolist(),
        'cache_note': 'FP16 probability caches; all reported single and ensemble results use the same decoded caches.',
        'test_note': 'Test labels are never read. Exported test masks have no measured test mIoU.',
        'selection_note': 'Validation scores may reflect repeated architecture/checkpoint selection; bootstrap does not correct that bias.',
        'runtime': {'device': str(device), 'torch': torch.__version__, 'smp': smp.__version__},
        'models': [],
    }
    code_hash = hashlib.sha256(''.join(file_sha256(Path(__file__).with_name(name))
                                     for name in ('train.py', 'metrics.py', 'dino_model.py', 'detail_refinement.py', 'native_unet.py', 'full_image.py')).encode()).hexdigest()
    model_caches, model_rows = [], []
    seen_checkpoints = set()
    for model_index, checkpoint_name in enumerate(args.checkpoints):
        checkpoint_path = Path(checkpoint_name).resolve()
        checkpoint_sha = file_sha256(checkpoint_path)
        if checkpoint_sha in seen_checkpoints:
            raise ValueError('Identical checkpoints supplied more than once; use explicit weights instead')
        seen_checkpoints.add(checkpoint_sha)
        # These are project-created local training checkpoints containing config as well as tensors.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        canonical_width = int(config.get('canonical_width', 0))
        whole_image = bool(config.get('full_image', False))
        if canonical_width < 0:
            raise ValueError('Checkpoint canonical_width must be nonnegative')
        settings = {'tile': args.tile, 'overlap': args.overlap, 'batch': args.batch,
                    'tta_d4': args.tta, 'canonical_width': canonical_width, 'whole_image': whole_image,
                    'device_type': device.type, 'torch': str(torch.__version__),
                    'smp': smp.__version__, 'code_sha256': code_hash}
        settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
        descriptor = {'index': model_index, 'checkpoint': str(checkpoint_path),
                      'checkpoint_sha256': checkpoint_sha, 'training_step': checkpoint.get('step'),
                      'config': config, 'inference': settings}
        rows, caches, model = [], [], None
        cache_hits = 0
        for sample in samples:
            metadata = {'schema_version': SCHEMA_VERSION, 'checkpoint_sha256': checkpoint_sha,
                        'image_id': sample['id'], 'image_sha256': sample['image_sha256'],
                        'shape': sample['shape'], 'inference': settings,
                        'class_names': CLASS_NAMES, 'palette': PALETTE.tolist()}
            cache_path = (cache_root / checkpoint_sha[:16] / settings_hash / args.split
                          / f"{sample['id']}_{sample['image_sha256'][:12]}.npz")
            probabilities = None
            if cache_path.exists() and not args.overwrite_cache:
                try:
                    probabilities = read_probability_cache(cache_path, metadata)
                    cache_hits += 1
                except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as error:
                    print(json.dumps({'invalid_cache': str(cache_path), 'reason': str(error)}), flush=True)
            if probabilities is None:
                if model is None:
                    model = build_model(config, pretrained=False)
                    model.load_state_dict(checkpoint['model'], strict=True)
                    model.to(device).eval()
                image = read_image(sample['path'])
                probabilities = tiled_probabilities(model, image, device, tile=args.tile,
                    overlap=args.overlap, batch=args.batch, tta=args.tta, canonical_width=canonical_width,whole_image=whole_image)
                write_probability_cache(cache_path, probabilities, metadata)
                # Use the same FP16 quantization for fresh and reused caches.
                probabilities = read_probability_cache(cache_path, metadata)
            caches.append((cache_path, metadata))
            if args.split == 'valid':
                rows.append(score_prediction(sample, probabilities.argmax(0).astype(np.uint8), read_image))
            print(json.dumps({'model': model_index, 'image': sample['id'], 'cached': str(cache_path),
                              'miou_absent_zero': rows[-1][METRIC_KEYS[0]] if rows else None}), flush=True)
            del probabilities
        descriptor['cache_hits'] = cache_hits
        descriptor['cache_files'] = [str(path.resolve()) for path, _ in caches]
        if rows:
            descriptor['validation'] = summarize_with_intervals(rows, args.bootstrap_samples, args.seed)
        report['models'].append(descriptor)
        model_caches.append(caches)
        model_rows.append(rows)
        del model, checkpoint
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        write_json(out / 'report.partial.json', report)

    equal_weights = normalize_weights(None, len(model_caches))
    selected_weights = normalize_weights(args.weights, len(model_caches))
    if args.optimize_weights:
        candidates = modest_simplex_candidates(len(model_caches))
        development = set(development_ids)
        score_table, ordered_ids = [], []
        for image_index, sample in enumerate(samples):
            if sample['id'] not in development:
                continue
            probabilities = [read_probability_cache(*caches[image_index]) for caches in model_caches]
            target = encode_mask(read_image(sample['mask']))
            candidate_scores = []
            for weights in candidates:
                prediction = weighted_probabilities(probabilities, weights).argmax(0).astype(np.uint8)
                row = score_confusion(confusion(target, prediction))
                candidate_scores.append([row[key] for key in METRIC_KEYS])
            score_table.append(candidate_scores)
            ordered_ids.append(sample['id'])
            del probabilities
        fit = fit_candidate_weights(score_table, candidates, ordered_ids, METRIC_KEYS.index(args.selection_metric))
        selected_weights = normalize_weights(fit['weights'], len(model_caches))
        report['development_weight_search'] = fit
        write_json(out / 'development_weight_search.json', fit)

    report['equal_weights'] = equal_weights.tolist()
    report['selected_weights'] = selected_weights.tolist()
    report['weight_source'] = 'development_fitted' if args.optimize_weights else 'explicit' if args.weights is not None else 'equal'
    same_weights = np.allclose(equal_weights, selected_weights, rtol=0, atol=1e-12)
    equal_rows, selected_rows, outputs = [], [], []
    for image_index, sample in enumerate(samples):
        probabilities = [read_probability_cache(*caches[image_index]) for caches in model_caches]
        equal_prediction = weighted_probabilities(probabilities, equal_weights).argmax(0).astype(np.uint8)
        prediction = equal_prediction if same_weights else weighted_probabilities(probabilities, selected_weights).argmax(0).astype(np.uint8)
        outputs.append(export_predictions(out / 'predictions' / args.split, sample, prediction))
        if args.split == 'valid':
            equal_rows.append(score_prediction(sample, equal_prediction, read_image))
            selected_rows.append(score_prediction(sample, prediction, read_image))
        print(json.dumps({'ensemble_image': sample['id'],
                          'miou_absent_zero': selected_rows[-1][METRIC_KEYS[0]] if selected_rows else None}), flush=True)
        del probabilities

    if args.split == 'valid':
        report['equal_weight_ensemble'] = summarize_with_intervals(equal_rows, args.bootstrap_samples, args.seed)
        report['selected_weight_ensemble'] = summarize_with_intervals(selected_rows, args.bootstrap_samples, args.seed)
        report['paired_delta_selected_minus_equal'] = paired_differences(equal_rows, selected_rows, args.bootstrap_samples, args.seed)
        report['paired_delta_selected_minus_each_single'] = [
            {'model_index': index, 'delta': paired_differences(rows, selected_rows, args.bootstrap_samples, args.seed)}
            for index, rows in enumerate(model_rows)
        ]
        if args.optimize_weights:
            development = set(development_ids)
            remaining = [row for row in selected_rows if row['id'] not in development]
            report['outside_weight_fitting_ids'] = [row['id'] for row in remaining]
            report['outside_weight_fitting'] = summarize_with_intervals(remaining, args.bootstrap_samples, args.seed) if remaining else None

    mapping = {'schema_version': SCHEMA_VERSION, 'split': args.split, 'class_names': CLASS_NAMES,
               'class_index_to_rgb': {str(index): color.tolist() for index, color in enumerate(PALETTE)},
               'weights_in_checkpoint_order': selected_weights.tolist(),
               'checkpoint_sha256_in_order': [model['checkpoint_sha256'] for model in report['models']],
               'tta_d4': args.tta, 'outputs': outputs,
               'note': 'RGB masks are submission candidates only. Index PNG values are class IDs 0..3, not rendered colors.'}
    write_json(out / 'mapping.json', mapping)
    if args.split == 'test':
        archive_path = out / 'submission_candidate_test.zip'
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for item in outputs:
                mask_path = Path(item['rgb_mask'])
                archive.write(mask_path, arcname=mask_path.name)
        report['submission_candidate_zip'] = str(archive_path.resolve())
        report['submission_candidate_note'] = 'Contains only RGB prediction PNGs; not uploaded or submitted. Complete contest submission also requires code, environment and proposal.'
        if len(samples) != 10:
            report['submission_candidate_warning'] = f'Official test set contains 10 images, but this export contains {len(samples)}.'
    report['mapping_file'] = str((out / 'mapping.json').resolve())
    write_json(out / 'report.json', report)
    write_json(out / 'weights.json', {'weights': selected_weights.tolist(),
                                     'checkpoints': [model['checkpoint'] for model in report['models']],
                                     'checkpoint_sha256': [model['checkpoint_sha256'] for model in report['models']],
                                     'source': report['weight_source']})
    (out / 'report.partial.json').unlink(missing_ok=True)
    print(json.dumps({'complete': True, 'report': str((out / 'report.json').resolve()),
                      'miou_absent_zero': report.get('selected_weight_ensemble', {}).get(METRIC_KEYS[0]),
                      'test_score_measured': False if args.split == 'test' else None}), flush=True)


if __name__ == '__main__':
    main()
