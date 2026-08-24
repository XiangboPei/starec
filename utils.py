import os
import random
import torch
import datetime
import argparse
import numpy as np
import logging


def set_logger(log_path, log_name="seqrec", mode="a"):
    """Set up log file (mode `a`=append / `w`=overwrite)."""
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def get_local_time():
    return datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")


def parse_args():
    parser = argparse.ArgumentParser(description="StaRec: sequential recommendation with spectral and attention branches")

    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="ML-1M", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument("--train_name", default=get_local_time(), type=str)

    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--gpu_id", default="0", type=str)

    parser.add_argument("--model_type", default="starec", choices=["starec", "duorec"], type=str.lower)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--num_hidden_layers", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument(
        "--contrastive_weight",
        default=0.1,
        type=float,
        help="DuoRec us_x contrastive-loss weight.",
    )
    parser.add_argument(
        "--contrastive_temperature",
        default=1.0,
        type=float,
        help="DuoRec InfoNCE temperature.",
    )

    # StaRec: default = fixed wavelet packet + self-attention; toggle branches with the two flags below.
    parser.add_argument("--alpha", default=0.9, type=float)
    parser.add_argument(
        "--wave",
        default="sym4",
        type=str,
        choices=["haar", "db2", "db4", "sym4", "sym6"],
        help="Orthogonal wavelet for the fixed packet spectral branch (when --use_spectral 1).",
    )
    parser.add_argument("--decomp_level", default=4, type=int, help="Wavelet packet depth when --use_spectral 1.")
    parser.add_argument(
        "--use_spectral",
        default=1,
        type=int,
        choices=[0, 1],
        help="1 (default): fixed wavelet packet branch in each sublayer; 0: turn spectral branch off (attention-only block if --use_attention 1).",
    )
    parser.add_argument(
        "--use_attention",
        default=1,
        type=int,
        choices=[0, 1],
        help="1 (default): self-attention branch in each sublayer; 0: no self-attention (wavelet-only block if --use_spectral 1).",
    )

    parsed_args = parser.parse_args()

    if parsed_args.model_type == "starec" and int(parsed_args.use_spectral) == 0 and int(parsed_args.use_attention) == 0:
        raise ValueError("StaRec: use_spectral and use_attention cannot both be 0 (empty sublayer).")
    if parsed_args.contrastive_weight < 0.0:
        raise ValueError("contrastive_weight cannot be negative.")
    if parsed_args.contrastive_temperature <= 0.0:
        raise ValueError("contrastive_temperature must be positive.")

    return parsed_args


class EarlyStopping:
    """Stop training when validation metric does not improve."""

    def __init__(self, checkpoint_path, logger, patience=10, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.logger = logger

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            self.logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            self.logger.info("Validation score increased. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score
