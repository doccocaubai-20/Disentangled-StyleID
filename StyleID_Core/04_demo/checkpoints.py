"""Resolve, verify, and cache the checkpoints needed by the selected demo mode."""
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import gdown


CORE_DIR = Path(__file__).resolve().parents[1]
SOUP_FILES = {
    'backbone': (
        'disentangled_lora_clip_SOUP_S6_A100.pt',
        '11RE2F-5JWI3jQymbJ5Z_QTpKSGKpwB6n',
        '45fbc2b7e09483c28da2d91d52dbca0114260cfb14e9277c6790b6d93819b24f',
    ),
    'head': (
        'disentangled_head_SOUP_S6_A100.pt',
        '1ifMm_3hwbYYb6Evr76T8FNtBjnQ69iK6',
        '5e9dddb7bdf0f45012abd127b83a31b77297b2f5c245cea52b878cfa71ac9968',
    ),
}
SOUP_SHA256 = {role: sha for role, (_, _, sha) in SOUP_FILES.items()}
ADAPTER_NAME = 'lfw_hard_positive_adapter.pt'
ADAPTER_FOLDER = 'hard-positive-adapter'
ADAPTER_DRIVE_ID = '12saHW4q3RPIVwel2uJpsnsgv_YJDZLA8'
ADAPTER_SHA256 = '363ae0e66630711a66a5836407a71d5f704317fd9824cdee409e8d44a1383c1e'


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected_hash):
    if file_sha256(path) != expected_hash:
        raise ValueError(f'SHA256 không khớp: {path}')
    return path


def download(file_id, target, expected_hash):
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, suffix='.partial', delete=False) as temporary:
        partial = Path(temporary.name)
    try:
        result = gdown.download(id=file_id, output=str(partial), quiet=False, timeout=(10, 60))
        if result is None:
            raise RuntimeError('Drive không trả về tệp')
        verify(partial, expected_hash)
        partial.replace(target)
    except Exception as exc:
        raise RuntimeError(f'Không tải được {target.name} từ Drive (ID {file_id}): {exc}') from exc
    finally:
        partial.unlink(missing_ok=True)
    return target


def soup_paths():
    cache = CORE_DIR / 'checkpoints' / 'soup'
    configured = os.environ.get('STYLEID_CHECKPOINT_DIR')
    candidates = [Path(configured).expanduser()] if configured else [cache, CORE_DIR, CORE_DIR.parent]
    paths = {}
    for role, (name, file_id, checksum) in SOUP_FILES.items():
        local = next((directory / name for directory in candidates if (directory / name).is_file()), None)
        paths[role] = verify(local, checksum) if local else download(file_id, candidates[0] / name, checksum)
    return paths['backbone'], paths['head']


def adapter_path():
    configured = os.environ.get('STYLEID_HP_ADAPTER_PATH')
    target = (Path(configured).expanduser() if configured
              else CORE_DIR / 'checkpoints' / ADAPTER_FOLDER / ADAPTER_NAME)
    if target.is_file():
        return verify(target, ADAPTER_SHA256)
    file_id = os.environ.get('STYLEID_HP_ADAPTER_DRIVE_ID', ADAPTER_DRIVE_ID)
    return download(file_id, target, ADAPTER_SHA256)
