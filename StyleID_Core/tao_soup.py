"""Create the historical 50/50 S6+A100 weight soup on CPU, only when run.

No training or benchmark. Existing canonical Soup files are not touched by default.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mix_states(first, second, weight):
    """Same float32 arithmetic then dtype cast as the original test_all.py."""
    import torch
    if set(first) != set(second):
        raise ValueError('Parent state dicts have different keys; no silent skipping')
    mixed = {}
    for key, a in first.items():
        b = second[key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError(f'Non-tensor entry: {key}')
        if a.shape != b.shape or a.dtype != b.dtype or a.is_complex() or b.is_complex():
            raise ValueError(f'Incompatible parent tensors: {key}')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f'Non-finite parent tensor: {key}')
        # Includes integer buffers, reproducing the legacy blend-and-cast rule.
        mixed[key] = (weight*a.float() + (1-weight)*b.float()).to(b.dtype)
        if not torch.isfinite(mixed[key]).all():
            raise ValueError(f'Invalid mixture tensor: {key}')
    return mixed


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir', type=Path, default=here.parent,
                        help='Directory containing the named shard6 and a100_final pairs')
    parser.add_argument('--shard6-backbone', type=Path)
    parser.add_argument('--shard6-head', type=Path)
    parser.add_argument('--a100-backbone', type=Path)
    parser.add_argument('--a100-head', type=Path)
    parser.add_argument('--weight-shard6', type=float, default=.5)
    parser.add_argument('--output-dir', type=Path, default=here/'checkpoints'/'soup')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.weight_shard6 <= 1:
        parser.error('--weight-shard6 must be between 0 and 1')
    inputs = {
        'shard6_backbone': args.shard6_backbone or args.parent_dir/'disentangled_lora_clip_shard6.pt',
        'shard6_head': args.shard6_head or args.parent_dir/'disentangled_head_shard6.pt',
        'a100_backbone': args.a100_backbone or args.parent_dir/'disentangled_lora_clip_a100_final.pt',
        'a100_head': args.a100_head or args.parent_dir/'disentangled_head_a100_final.pt',
    }
    inputs = {k: p.resolve() for k, p in inputs.items()}
    for p in inputs.values():
        if not p.is_file():
            parser.error(f'Missing parent checkpoint: {p}')
    output = args.output_dir.resolve()
    destinations = {
        'backbone': output/'disentangled_lora_clip_SOUP_S6_A100.pt',
        'head': output/'disentangled_head_SOUP_S6_A100.pt',
    }
    for p in [*destinations.values(), output/'soup_manifest.json']:
        if p in inputs.values():
            parser.error('Output cannot replace a parent checkpoint')
        if p.exists() and not args.overwrite:
            parser.error(f'Output exists: {p}. Use a different --output-dir or explicit --overwrite.')
    import torch
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'method': 'float32 weighted mean then cast to original dtype, including integer buffers',
                'weight_shard6': args.weight_shard6, 'weight_a100': 1-args.weight_shard6,
                'input_sha256': {k: sha256(p) for k, p in inputs.items()},
                'input_paths': {k: str(p) for k, p in inputs.items()}, 'device': 'cpu', 'outputs': {}}
    pending = []
    try:
        for kind, target in destinations.items():
            print(f'Mixing {kind} on CPU...', flush=True)
            a = torch.load(inputs[f'shard6_{kind}'], map_location='cpu', weights_only=True)
            b = torch.load(inputs[f'a100_{kind}'], map_location='cpu', weights_only=True)
            result = mix_states(a, b, args.weight_shard6)
            temp = target.with_suffix('.pt.pending')
            pending.append(temp)
            torch.save(result, temp)
            metadata['outputs'][kind] = {'path': str(target), 'sha256': sha256(temp), 'tensors': len(result)}
            del a, b, result
            gc.collect()
        for kind, target in destinations.items():
            target.with_suffix('.pt.pending').replace(target)
        (output/'soup_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    finally:
        for temp in pending:
            if temp.exists():
                temp.unlink()
    print(f'Saved Soup and provenance in: {output}')


if __name__ == '__main__':
    main()
