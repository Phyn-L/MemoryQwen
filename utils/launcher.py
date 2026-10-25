"""One worker per visible GPU; experiment parameters belong to YAML, not shell."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {'train': 'train.py', 'test': 'test.py'}


def command(mode, args, device_count):
    script = str(ROOT / 'scripts' / SCRIPTS[mode])
    if any(arg in ('-h', '--help') for arg in args):
        return [sys.executable, script, *args]
    count = int(os.environ.get('NUM_PROCESSES', str(max(1, device_count))))
    if count < 1 or (count > 1 and count > device_count):
        raise ValueError(f'NUM_PROCESSES={count}, but only {device_count} CUDA devices are visible')
    if count == 1:
        return [sys.executable, script, *args]
    if count > 1:
        import importlib.util
        if importlib.util.find_spec('accelerate') is None:
            raise RuntimeError('Multi-GPU train/test requires accelerate in the active Python environment')
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc-per-node={count}', script, *args]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('mode', choices=SCRIPTS)
    mode, args = parser.parse_known_args()
    if mode.mode == 'train':
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if any(a in ('-h', '--help') for a in args):
        count = 0
    else:
        import torch
        count = torch.cuda.device_count()
    cmd = command(mode.mode, args, count)
    os.execv(sys.executable, cmd)


if __name__ == '__main__':
    main()
