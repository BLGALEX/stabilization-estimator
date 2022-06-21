import os
from pathlib import Path


BASE_DIR = Path(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

MODEL_PATHES = {
    'seresnet18': BASE_DIR / 'models/seresnet18',
    'resnet50': BASE_DIR / 'models/seresnet18',
}
