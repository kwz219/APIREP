from apirep.Utils.Logging import logger
import torch

def load_checkpoint(ptpath):
    if ptpath:
        logger.info('Loading pretrained model from %s'% ptpath)
        return torch.load(ptpath,map_location=lambda storage,loc:storage)

def load_seq2seq(path):
    if path:
        loaded_seq2seq=load_checkpoint(ptpath=path)
    else:
        raise ValueError("you must assign a path of the pretrained seq2seq model")
