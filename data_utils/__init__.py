"""
data_utils - A collection of utilities for data management, processing and save.
"""

from .pickle_utils import save_as_pickle, load_from_pickle, load_pickles_from_directory

__all__ = [
    "save_as_pickle",
    "load_from_pickle",
    "load_pickles_from_directory"
]

__version__ = "0.0.1"
