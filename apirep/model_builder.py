import re

from onmt.model_builder import build_encoder, build_decoder
from onmt.models import NMTModel
from onmt.modules import Embeddings
import torch.nn as nn
from apirep.Utils.Logging import logger
import torch

def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    freeze_word_vecs = opt.freeze_word_vecs_enc if for_encoder \
        else opt.freeze_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        freeze_word_vecs=freeze_word_vecs
    )
    return emb

def build_src_emb(model_opt, fields):
    # Build embeddings.
    if model_opt.model_type == "text":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None
    return src_emb


def build_encoder_with_embeddings(model_opt, fields):
    # Build encoder.
    src_emb = build_src_emb(model_opt, fields)
    encoder = build_encoder(model_opt, src_emb)
    return encoder, src_emb


def build_decoder_with_embeddings(
    model_opt, fields, share_embeddings=False, src_emb=None
):
    # Build embeddings.
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

    if share_embeddings:
        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    # Build decoder.
    decoder = build_decoder(model_opt, tgt_emb)
    return decoder, tgt_emb

def build_model(model_opt,opt,checkpoint,fields):
    logger.info("building model.....")

    #0. Set device
    if opt.gpu and opt.gpu_id is not None:
        device = torch.device("cuda", opt.gpu_id)
    elif opt.gpu and not opt.gpu_id:
        device = torch.device("cuda")
    elif not opt.gpu:
        device = torch.device("cpu")

    #1.Build a seq2seq Model containing encoder and decoder
    encoder, src_emb = build_encoder_with_embeddings(model_opt, fields)
    decoder, _ = build_decoder_with_embeddings(
        model_opt,
        fields,
        share_embeddings=model_opt.share_embeddings,
        src_emb=src_emb,
    )
    model=NMTModel(encoder=encoder,decoder=decoder)
    logger.info("encoder type %s,decoder type %s",type(model.encoder),type(model.decoder))

    #2.Build a generator (for finetune)
    generator=nn.Sequential(nn.Linear(model_opt.dec_rnn_size,opt.n_labels)
                            )#先用最简单的线性层试一下

    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s
        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        model.load_state_dict(checkpoint['model'], strict=False)

    model.generator=generator
    model.to(device)
    return model


