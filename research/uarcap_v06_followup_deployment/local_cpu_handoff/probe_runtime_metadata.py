"""Print a version-and-origin receipt with stdlib only. No packages or model are imported."""

from hashlib import sha256
from importlib import metadata
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import sys

PACKAGES = (
    ('torch','torch'), ('torchvision','torchvision'), ('transformers','transformers'), ('timm','timm'),
    ('numpy','numpy'), ('Pillow','PIL'), ('opencv-python','cv2'), ('av','av'), ('PyYAML','yaml'),
    ('einops','einops'), ('tokenizers','tokenizers'), ('sentencepiece','sentencepiece'),
    ('huggingface-hub','huggingface_hub'), ('accelerate','accelerate'), ('safetensors','safetensors'),
)
PATH_ENV = ('LD_LIBRARY_PATH','LD_PRELOAD','CONDA_PREFIX','VIRTUAL_ENV','PYTHONPATH','PYTHONHOME',
            'TORCH_HOME','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_OFFLINE','HF_HUB_OFFLINE')
SMALL_FILE_LIMIT = 8 * 1024 * 1024


def package_receipt(distribution_name, import_name):
    try:
        dist = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        dist = None
    spec = PathFinder.find_spec(import_name)
    row = {
        'version':dist.version if dist else None,
        'distribution_root':str(dist.locate_file('')) if dist else None,
        'module_origin':spec.origin if spec else None,
        'metadata_text_sha256':{},
    }
    if dist:
        for descriptor in ('METADATA','RECORD','INSTALLER'):
            value = dist.read_text(descriptor)
            if value is not None:
                row['metadata_text_sha256'][descriptor] = sha256(value.encode('utf-8')).hexdigest()
    if spec and spec.origin:
        origin = Path(spec.origin)
        if origin.is_file() and origin.stat().st_size <= SMALL_FILE_LIMIT:
            row['module_file_sha256'] = sha256(origin.read_bytes()).hexdigest()
    return row


def snapshot():
    return {
        'interpreter':sys.executable, 'real_interpreter':str(Path(sys.executable).resolve()),
        'version':sys.version, 'prefix':sys.prefix, 'base_prefix':sys.base_prefix,
        'isolated':bool(sys.flags.isolated), 'ignore_python_environment':bool(sys.flags.ignore_environment),
        'no_user_site':bool(sys.flags.no_user_site), 'sys_path':sys.path,
        'environment':{name:os.environ[name] for name in PATH_ENV if name in os.environ},
        'packages':{dist:package_receipt(dist, module) for dist,module in PACKAGES},
        'loads_runtime_packages':False,
    }


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2, sort_keys=True))
