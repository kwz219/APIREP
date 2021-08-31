from onmt.models import build_model_saver
from onmt.utils import Optimizer
from apirep.Inputters.dynamic_iterator import build_dynamic_dataset_iter
from apirep.Utils.ArgumentParser import ArgumentParser
from apirep.Utils.Logging import logger
from apirep.model_builder import build_model
from apirep.trainer import build_trainer
import torch

class IterOnDevice(object):
    """Sent items from `iterable` on `device_id` and yield."""

    def __init__(self, iterable, device_id):
        self.iterable = iterable
        self.device_id = device_id

    @staticmethod
    def batch_to_device(batch, device_id):
        """Move `batch` to `device_id`, cpu if `device_id` < 0."""
        curr_device = batch.indices.device
        device = torch.device(device_id) if device_id >= 0 \
            else torch.device('cpu')
        if curr_device != device:
            if isinstance(batch.src, tuple):
                batch.src = tuple([_.to(device) for _ in batch.src])
            else:
                batch.src = batch.src.to(device)
            batch.tgt = batch.tgt.to(device)
            batch.indices = batch.indices.to(device)
            batch.alignment = batch.alignment.to(device) \
                if hasattr(batch, 'alignment') else None
            batch.src_map = batch.src_map.to(device) \
                if hasattr(batch, 'src_map') else None
            batch.align = batch.align.to(device) \
                if hasattr(batch, 'align') else None

    def __iter__(self):
        for batch in self.iterable:
            self.batch_to_device(batch, self.device_id)
            yield batch
def _build_train_iter(opt, fields, transforms_cls, stride=1, offset=0):
    """Build training iterator."""
    train_iter = build_dynamic_dataset_iter(
        fields, transforms_cls, opt, is_train=True,
        stride=stride, offset=offset)
    return train_iter

def _build_valid_iter(opt, fields, transforms_cls):
    """Build iterator used for validation."""
    valid_iter = build_dynamic_dataset_iter(
        fields, transforms_cls, opt, is_train=False)
    return valid_iter

def _get_model_opts(opt, checkpoint=None):
    """Get `model_opt` to build model, may load from `checkpoint` if any."""
    if checkpoint is not None:
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        if (opt.tensorboard_log_dir == model_opt.tensorboard_log_dir and
                hasattr(model_opt, 'tensorboard_log_dir_dated')):
            # ensure tensorboard output is written in the directory
            # of previous checkpoints
            opt.tensorboard_log_dir_dated = model_opt.tensorboard_log_dir_dated
        # Override checkpoint's update_embeddings as it defaults to false
        model_opt.update_vocab = opt.update_vocab
    else:
        model_opt = opt
    return model_opt
def train(opt,checkpoint,device_id,fields):
    model_opt=_get_model_opts(opt,checkpoint=checkpoint)

    #Build model, containing pretrained seq2seq and tokenCLSLayer
    model=build_model(model_opt,opt,checkpoint)
    model.count_parameters(log=logger.info)

    #Build Optimizer
    optim=Optimizer.from_opt(model,opt,checkpoint=checkpoint)

    #Build model_saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    #Build trainer
    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver)


    _train_iter = _build_train_iter(opt, fields, transforms_cls=None)#TODO 这里可能会出bug
    train_iter = IterOnDevice(_train_iter, device_id)

    valid_iter = _build_valid_iter(opt, fields, transforms_cls=None)
    train_steps = opt.train_steps

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)