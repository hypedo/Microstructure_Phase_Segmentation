import contextlib
import hashlib
import json
import os
import platform
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
import torchvision


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check server PyTorch build and GPU driver')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    return torch.device(name)


def environment(device):
    dependencies = {}
    for package in ('numpy', 'Pillow', 'timm', 'mamba-ssm', 'causal-conv1d'):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__), 'cuda_build': torch.version.cuda,
            'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'dependencies': dependencies, 'code_sha256': code_hash()}


def code_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_checkpoint(path, state):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(cpu_tree(state), temp)
    temp.replace(path)


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def capture_rng(generator=None):
    numpy_state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'mps': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            'loader': generator.get_state() if generator is not None else None}


def restore_rng(state, generator=None):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed; exact RNG resume unavailable')
        torch.cuda.set_rng_state_all(state['cuda'])
    if state.get('mps') is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])
    if generator is not None:
        generator.set_state(state['loader'])


@contextlib.contextmanager
def run_directory(path, resume=False):
    # Linux/macOS advisory lock automatically releases on crash; no stale PID lock.
    import fcntl
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process is using {path}') from exc
        files = [p for p in path.iterdir() if p.name != '.run.lock']
        if files and not resume:
            raise ValueError(f'Output is nonempty: {path}; choose a new directory or --resume')
        if resume and not (path / 'last.pt').is_file():
            raise ValueError('--resume requires last.pt')
        yield path


def resume_guard(checkpoint, config, data_manifest, phase, device):
    if checkpoint.get('phase') != phase or checkpoint['config'] != config:
        raise ValueError('Resume phase/config differs; epochs and all settings must match')
    if checkpoint['data_manifest'] != data_manifest or checkpoint['environment']['code_sha256'] != code_hash():
        raise ValueError('Resume dataset or code differs; do not mix runs')
    current = environment(device)
    for key in ('device', 'torch', 'torchvision', 'cuda_build', 'gpu', 'dependencies'):
        if checkpoint['environment'].get(key) != current[key]:
            raise ValueError(f'Resume environment differs at {key}; evaluate checkpoint or start a separate run')
