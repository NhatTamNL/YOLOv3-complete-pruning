import torch
from models import *
from utils.utils import *
from copy import deepcopy
from utils.prune_utils import *

class opt():
    model_def = "normal_prune_0.3_yolov3-tiny.cfg"  # file config prune at repo Model_Compression_For_YOLOV3-V4
    data_config = "data_firev6/yolo.data" 
    model = 'weights/best.pt' # file train Model_Compression_For_YOLOV3-V4

percent = 0.3
num_filters = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(opt.model_def).to(device)

CBL_idx, Conv_idx, prune_idx= parse_module_defs(model.module_defs)

compact_module_defs = deepcopy(model.module_defs)
for idx, num in zip(CBL_idx, num_filters):
    assert compact_module_defs[idx]['type'] == 'convolutional'
    compact_module_defs[idx]['filters'] = str(num)

compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs).to(device)
compact_model_name = 'weights/yolov3_firev2_normal_pruning_'+str(percent)+'percent.weights'

save_weights(compact_model, path=compact_model_name)
print(f'Compact model has been saved: {compact_model_name}')