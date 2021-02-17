"""
Description of psmatching
"""

from psmatching import match
from psmatching import utilities
import os.path as _osp

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')
