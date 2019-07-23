# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import sys
sys.path.append('.')
from data import get_data_bunch
from config import cfg


if __name__ == '__main__':
    data = get_data_bunch(cfg)
    from IPython import embed; embed()