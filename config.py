import os

import torch

datasets_path = os.getenv("DATASETS_PATH", ".")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
