import os
from pathlib import Path


BASE_DIR = Path(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

MODEL_PATHES = {
    'resnet50': BASE_DIR / 'models/resnet50',
}
