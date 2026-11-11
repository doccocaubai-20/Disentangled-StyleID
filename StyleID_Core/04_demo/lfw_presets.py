"""Serve a small selection of pairs from the benchmark's LFW View 2 split."""
import os
from pathlib import Path
from shutil import copyfileobj
import tarfile
from tempfile import TemporaryDirectory
from threading import Lock
from urllib.request import urlopen

from checkpoints import CORE_DIR, verify


LFW_URL = 'https://ndownloader.figshare.com/files/5976015'
LFW_SHA256 = 'b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a'
PRESET_DIR = CORE_DIR / 'data' / 'lfw-demo'
PRESETS = {
    'lfw-1342': (
        'Cùng người · một ảnh đeo kính râm',
        True,
        ('Keanu_Reeves/Keanu_Reeves_0006.jpg', 'Keanu_Reeves/Keanu_Reeves_0009.jpg'),
    ),
    'lfw-0557': (
        'Khác người · cặp khó',
        False,
        ('Mark_Everson/Mark_Everson_0001.jpg', 'Martin_Sheen/Martin_Sheen_0002.jpg'),
    ),
    'lfw-0000': (
        'Cùng người · cặp đối chứng',
        True,
        ('Abel_Pacheco/Abel_Pacheco_0001.jpg', 'Abel_Pacheco/Abel_Pacheco_0004.jpg'),
    ),
}
PRESET_IMAGES = {name for _, _, names in PRESETS.values() for name in names}
PRESET_LOCK = Lock()


def extract_presets(archive_path, output):
    verify(archive_path, LFW_SHA256)
    remaining = PRESET_IMAGES.copy()
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            key = '/'.join(member.name.split('/')[-2:])
            if not member.isfile() or key not in remaining:
                continue
            target = output / key
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as destination:
                copyfileobj(source, destination)
            remaining.remove(key)
            if not remaining:
                return
    raise FileNotFoundError(f'Archive LFW thiếu ảnh: {", ".join(sorted(remaining))}')


def preset_paths(preset_id):
    _, _, names = PRESETS[preset_id]
    paths = tuple(PRESET_DIR / name for name in names)
    with PRESET_LOCK:
        if all(path.is_file() for path in paths):
            return paths
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=PRESET_DIR) as temporary:
            staging = Path(temporary)
            configured = os.environ.get('STYLEID_LFW_ARCHIVE')
            if configured:
                archive = Path(configured).expanduser()
            else:
                archive = staging / 'lfw-funneled.tgz'
                with urlopen(LFW_URL, timeout=60) as response, archive.open('wb') as destination:
                    copyfileobj(response, destination)
            extract_presets(archive, staging)
            for name in PRESET_IMAGES:
                target = PRESET_DIR / name
                target.parent.mkdir(parents=True, exist_ok=True)
                (staging / name).replace(target)
    return paths
