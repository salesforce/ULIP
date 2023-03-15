import sys
import os
pointnext_dir = './models/pointnext/PointNeXt'

def add_path_recursive(directory):
    sys.path.append(directory)
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            add_path_recursive(os.path.join(root, d))

add_path_recursive(pointnext_dir)

# print(sys.path)
from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg
from openpoints.utils import cal_model_parm_nums

def PointNEXT():
    cfg_path = './models/pointnext/pointnext-s.yaml'

    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)

    model = build_model_from_cfg(cfg.model)
    model_size = cal_model_parm_nums(model)
    print("model size:")
    print(model_size)

    return model