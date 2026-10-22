"""Validated NumPy post-filter for the frozen Soup demo (no model loading)."""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


class DemoLeace:
    def __init__(self, operator_path, metadata_path):
        metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
        if file_sha256(operator_path) != metadata['operator_sha256']:
            raise ValueError('LEACE operator checksum mismatch')
        with np.load(operator_path, allow_pickle=False) as data:
            if set(data.files) != {'mean','left','right'}:
                raise ValueError('Unexpected LEACE fields')
            self.mean, self.left, self.right = [data[k].astype(np.float64) for k in ('mean','left','right')]
        rank = metadata['rank']
        if self.mean.shape!=(512,) or self.left.shape!=(512,rank) or self.right.shape!=(rank,512):
            raise ValueError('Invalid Soup LEACE dimensions')
        if not all(np.isfinite(x).all() for x in (self.mean,self.left,self.right)):
            raise ValueError('Nonfinite LEACE operator')
        self.metadata=metadata

    def validate_checkpoints(self, backbone, head):
        for name,path in [('backbone',backbone),('head',head)]:
            if file_sha256(path)!=self.metadata['checkpoint_sha256'][name]:
                raise ValueError(f'LEACE requires the frozen Soup {name}; checkpoint hash mismatch')

    def apply(self, identity):
        x=np.asarray(identity,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=512 or not np.isfinite(x).all():
            raise ValueError('Expected finite ID512 vectors')
        y=x-((x-self.mean)@self.right.T)@self.left.T
        norms=np.linalg.norm(y,axis=1,keepdims=True)
        if (norms<1e-12).any() or not np.isfinite(y).all():
            raise ValueError('Invalid LEACE output')
        return y/norms
