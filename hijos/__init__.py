import os

from madre import import_modules

PATHS = [f"{os.path.dirname(os.path.realpath(__file__))}"]
import_modules(PATHS)

__version__ = "0.0.1"
