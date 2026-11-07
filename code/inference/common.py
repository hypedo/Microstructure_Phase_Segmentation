"""Manifest and probability contracts; no Torch or training code imports."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import re

CLASSES = ['Al', 'Primary_Si', 'Al3Ni', 'Eutectic_Si']
PALETTE = [[153, 127, 76], [76, 178, 76], [25, 76, 153], [204, 204, 204]]
COMPOSITE_FILES = ['train.py', 'metrics.py', 'dino_model.py', 'detail_refinement.py', 'native_unet.py']
INFERENCE_FIELDS = {'tile', 'overlap', 'batch', 'tta_d4', 'canonical_width', 'whole_image',
    'cudnn_benchmark', 'source_sha256', 'code_sha256', 'torch', 'smp', 'device_type',
    'actual_canonical_width_override', 'checkpoint_canonical_width', 'config_mutated'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError(f'Non-finite JSON value: {value}')
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=bad_constant)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def resolve_file(root, relative):
    require(isinstance(relative, str), 'Bundle paths must be strings')
    p = PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == relative,
            f'Unsafe or noncanonical bundle path: {relative}')
    path = Path(root).joinpath(*p.parts)
    for parent in [path, *path.parents]:
        if parent == Path(root).parent:
            break
        require(not parent.is_symlink(), f'Symlinks are not allowed: {path}')
    require(path.is_file(), f'Missing bundle file: {path}')
    return path


def inference_settings(value):
    require(isinstance(value, dict), 'Each model requires an inference object')
    require(not set(value)-INFERENCE_FIELDS, f'Unsupported inference fields: {sorted(set(value)-INFERENCE_FIELDS)}')
    whole = value.get('whole_image', False)
    for key in ['batch']:
        require(type(value.get(key)) is int, f'Explicit integer inference.{key} is required')
    if whole:
        require('tile' in value and 'overlap' in value, 'Record tile/overlap explicitly even when unused')
        require(value['tile'] is None and value['overlap'] is None
                or type(value['tile']) is int and type(value['overlap']) is int,
                'Whole-image tile/overlap must be explicit nulls or recorded integers')
    else:
        require(type(value.get('tile')) is int and type(value.get('overlap')) is int,
                'Tiled inference requires explicit integer tile/overlap')
        require(value['tile'] > value['overlap'] >= 0 and value['tile'] % 32 == 0,
                'Require tile divisible by32 and 0 <= overlap < tile')
    require(value['batch'] >= 1, 'Inference batch must be positive')
    require(type(value.get('tta_d4')) is bool, 'Explicit boolean tta_d4 is required')
    canonical = value.get('actual_canonical_width_override', value.get('canonical_width'))
    require(type(canonical) is int and canonical >= 0, 'Explicit nonnegative canonical width is required')
    if 'actual_canonical_width_override' in value and 'canonical_width' in value:
        require(value['canonical_width'] == canonical, 'Conflicting canonical inference overrides')
    for key in ['cudnn_benchmark', 'whole_image', 'config_mutated']:
        if key in value:
            require(type(value[key]) is bool, f'{key} must be boolean')
    require(value.get('config_mutated', False) is False, 'Checkpoint config mutation is not supported')
    require(value.get('cudnn_benchmark', False) is False,
            'This runner requires the recorded/default fresh-process cudnn_benchmark=False')
    require('source_sha256' in value or 'code_sha256' in value, 'Frozen inference source identity is required')
    require(isinstance(value.get('torch'), str), 'Recorded Torch version is required')
    return {'tile': value['tile'], 'overlap': value['overlap'], 'batch': value['batch'],
            'tta': value['tta_d4'], 'canonical_width': canonical,
            'whole_image': value.get('whole_image', False), 'cudnn_benchmark': False}


def validate_manifest(manifest):
    require(isinstance(manifest, dict), 'Manifest must be an object')
    require(manifest.get('kind') == 'fixed_probability_ensemble', 'Only fixed probability ensembles are supported')
    order, models, weights = [manifest.get(k) for k in ['model_order', 'models', 'weights']]
    require(isinstance(order, list) and order and all(isinstance(k, str) and re.fullmatch('[A-Za-z0-9_-]+', k) for k in order),
            'model_order must contain safe nonempty model names')
    require(len(set(order)) == len(order), 'Duplicate model names')
    require(isinstance(models, dict) and set(order) == set(models), 'model_order and models must match exactly')
    require(isinstance(weights, list) and len(weights) == len(order), 'One fixed weight per model is required')
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in weights) and any(weights),
            'Weights must be finite, nonnegative and have positive total')
    require(abs(math.fsum(weights)-1) <= 1e-12, 'Weights must already sum to1; fitting/implicit normalization is forbidden')
    for name in order:
        model = models[name]
        require(isinstance(model, dict), f'Invalid model record: {name}')
        require(not set(model)-{'checkpoint_sha256', 'inference'}, f'Unsupported model fields: {name}')
        require(valid_sha(model.get('checkpoint_sha256')), f'Missing/invalid checkpoint hash: {name}')
        inference_settings(model['inference'])
    return manifest


def verify_source(source_files, inference):
    expected = inference.get('source_sha256')
    if expected is not None:
        require(isinstance(expected, dict) and expected, 'source_sha256 must be a nonempty map')
        for name, value in expected.items():
            require(valid_sha(value) and source_files.get(name) == value, f'Frozen source hash mismatch: {name}')
    if 'code_sha256' in inference:
        require(all(name in source_files for name in COMPOSITE_FILES), 'Composite source hash is missing required modules')
        combined = hashlib.sha256(''.join(source_files[name] for name in COMPOSITE_FILES).encode()).hexdigest()
        require(combined == inference['code_sha256'], 'Composite inference code_sha256 mismatch')


def verify_bundle(model_dir, manifest_path=None):
    root = Path(model_dir).resolve()
    index = load_json(root/'bundle.json')
    require(set(index) == {'schema_version', 'manifest_sha256', 'class_names', 'palette', 'models'}, 'Unknown/missing bundle fields')
    require(index['schema_version'] == 1 and index['class_names'] == CLASSES and index['palette'] == PALETTE,
            'Unsupported bundle schema/classes/palette')
    bundled_manifest = resolve_file(root, 'current_best.json')
    require(sha(bundled_manifest) == index['manifest_sha256'], 'Bundled manifest SHA mismatch')
    if manifest_path:
        require(sha(manifest_path) == index['manifest_sha256'], 'Supplied manifest differs from frozen bundle')
    manifest = validate_manifest(load_json(bundled_manifest))
    require(set(index['models']) == set(manifest['models']), 'Bundle model names do not match manifest')
    allowed = {'bundle.json', 'current_best.json'}
    records = {}
    for name in manifest['model_order']:
        spec = index['models'][name]
        require(set(spec) == {'checkpoint', 'config', 'config_sha256', 'source_dir', 'source_sha256'},
                f'Unknown/missing model binding fields: {name}')
        checkpoint = resolve_file(root, spec['checkpoint'])
        config = resolve_file(root, spec['config'])
        require(sha(checkpoint) == manifest['models'][name]['checkpoint_sha256'], f'Checkpoint SHA mismatch: {name}')
        require(sha(config) == spec['config_sha256'], f'Config SHA mismatch: {name}')
        require(isinstance(spec['source_sha256'], dict) and spec['source_sha256'], f'Missing source map: {name}')
        source_files = {}
        for filename, expected in spec['source_sha256'].items():
            require(re.fullmatch('[A-Za-z0-9_]+\\.py', filename) is not None, f'Invalid source filename: {filename}')
            relative = spec['source_dir']+'/'+filename
            path = resolve_file(root, relative)
            actual = sha(path)
            require(valid_sha(expected) and actual == expected, f'Bundle source changed: {relative}')
            source_files[filename] = actual
            allowed.add(relative)
        require({'train.py', 'metrics.py'}.issubset(source_files), f'Required source module missing: {name}')
        verify_source(source_files, manifest['models'][name]['inference'])
        allowed.update([spec['checkpoint'], spec['config']])
        records[name] = {'checkpoint': checkpoint, 'config': config, 'source_dir': root/spec['source_dir'],
                         'source_sha256': source_files, 'settings': inference_settings(manifest['models'][name]['inference'])}
    actual = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), f'Symlink is not allowed: {path}')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == allowed, f'Unknown/missing bundle files: {sorted(actual ^ allowed)}')
    return manifest, index, records


def input_inventory(directory):
    from PIL import Image
    root = Path(directory).resolve()
    require(root.is_dir() and root.name.lower() not in {'masks', 'labels', 'annotations'}, 'Provide an image directory, not labels')
    rows, seen = [], set()
    for path in sorted(root.iterdir()):
        require(path.is_file() and not path.is_symlink(), f'Unknown input entry (flat image directory required): {path.name}')
        require(path.suffix.lower() == '.png', f'Unknown input file: {path.name}; only original PNG images are supported')
        stem = path.stem.removesuffix('_image')
        require(stem and not re.search(r'(^|_)(mask|label|prediction|annotation)s?$', stem, re.I), f'Label/output filename rejected: {path.name}')
        require(stem.casefold() not in seen, f'Duplicate normalized image stem: {stem}')
        seen.add(stem.casefold())
        with Image.open(path) as im:
            require(im.format == 'PNG' and im.mode in {'RGB','RGBA','L'} and getattr(im, 'n_frames', 1) == 1,
                    f'Input must be a single 8-bit RGB, opaque RGBA, or grayscale PNG: {path.name}')
            if im.mode == 'RGBA':
                require(im.getchannel('A').getextrema() == (255,255), f'Transparent input is not supported: {path.name}')
            shape = [im.height, im.width]
        # Alpha inspection loads pixels; PNG.verify requires a fresh decoder.
        with Image.open(path) as verify_image:
            verify_image.verify()
        rows.append({'id': stem, 'path': str(path), 'sha256': sha(path), 'shape': shape})
    require(rows, 'No input PNG images found')
    return rows


def normalize_probabilities(probabilities, shape):
    import numpy as np
    p = np.asarray(probabilities)
    require(p.shape == (4, *shape), f'Probability geometry mismatch: {p.shape} versus {(4, *shape)}')
    require(np.issubdtype(p.dtype, np.floating) and np.isfinite(p).all() and p.min() >= -1e-5 and p.max() <= 1.001,
            'Invalid probability values')
    p = p.astype(np.float32)
    totals = p.sum(axis=0)
    require(np.allclose(totals, 1, rtol=0, atol=.005), 'Probabilities do not sum to1')
    p /= totals[None]
    return p


def cache_probabilities(raw, shape):
    import numpy as np
    return normalize_probabilities(raw, shape).astype(np.float16)


def aggregate(stored_probabilities, weights, shape):
    import numpy as np
    require(len(stored_probabilities) == len(weights) and len(weights) > 0, 'Weight/array count mismatch')
    require(all(type(w) in (int, float) and math.isfinite(w) and w > 0 for w in weights)
            and abs(math.fsum(weights)-1) <= 1e-12, 'Invalid fixed weights')
    # Match the original scorer's float64 normalization, limited to roundoff.
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    output = np.zeros((4, *shape), np.float32)
    for stored, weight in zip(stored_probabilities, weights):
        require(stored.dtype == np.float16, 'Aggregation requires actual FP16 model caches')
        output += normalize_probabilities(stored, shape)*np.float32(weight)
    return output


def verify_release(root):
    root=Path(root).resolve();index=load_json(root/'FILES.json')
    for relative,expected in index['files'].items():
        require(sha(resolve_file(root,relative))==expected,'Release file hash mismatch: '+relative)
    manifest=load_json(root/'manifest.json')
    for name,expected in manifest['runtime_source_sha256'].items():
        require(Path(name).name==name and sha(Path(__file__).parent/name)==expected,'Runtime source hash mismatch: '+name)
    require(manifest['class_names']==CLASSES and manifest['palette']==PALETTE,'Class/palette mismatch')
    names=[m['name']for m in manifest['models']]
    require(len(names)==len(set(names)) and len(names)==len(manifest['weights']),'Model order mismatch')
    weights=manifest['weights'];require(all(isinstance(w,(float,int)) and math.isfinite(w) and w>=0 for w in weights) and sum(weights)>0,'Invalid weights')
    settings=manifest.get('postprocess',{})
    require(not set(settings)-{'log_bias','primary_component_max_pixels','primary_presence_min_fraction','conditional_calibration'},'Unsupported postprocess setting')
    if 'log_bias'in settings:
        b=settings['log_bias'];require(isinstance(b,list) and len(b)==4 and b[0]==0 and all(isinstance(v,(float,int)) and math.isfinite(v) and abs(v)<=.35 for v in b),'Invalid class offsets')
    if 'conditional_calibration' in settings:
        from postprocess import validate_calibration
        validate_calibration(settings['conditional_calibration'])
    t=settings.get('primary_component_max_pixels',0);require(type(t)is int and 0<=t<=256,'Invalid component threshold')
    t=settings.get('primary_presence_min_fraction',0.);require(isinstance(t,(float,int)) and math.isfinite(t) and 0<=t<=1,'Invalid presence threshold')
    return manifest,index
