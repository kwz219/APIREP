from apirep.Utils.LoadPretrain import load_checkpoint
from apirep.model_builder import build_model


class opt(object):
    def __init__(self):
        self.gpu_ranks=0
        self.gpu=True
        self.gpu_id=0


class modelopt(object):
    def __init__(self):
        self.gpu_ranks = 0
def test_loadcheckpoint(filepath):
    file_path = ""
    checkpoint = load_checkpoint(file_path)
    print(checkpoint)
    print(checkpoint["opt"])
def test_loadpretrain(file_path,modelopt,opt):
    checkpoint=load_checkpoint(file_path)
    print(checkpoint)
    print(checkpoint["opt"])
    model = build_model(modelopt, opt, checkpoint)
    print(model.count_parameters())

