import os
import time

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from dataset import get_dataloder, get_rating_matrix, get_seq_dic
from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, parse_args, set_logger, set_seed


def main():
    args = parse_args()
    check_path(args.output_dir)
    log_path = os.path.join(args.output_dir, args.train_name + ".log")
    logger = set_logger(log_path, log_name=f"seqrec.{args.train_name}", mode="w")

    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + ".pt")
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args, seq_dic)

    logger.info(str(args))
    logger.info("Model: %s", args.model_type)
    if args.model_type == "starec":
        logger.info(
            "StaRec config | use_spectral=%s use_attention=%s alpha=%.4f wave=%s decomp_level=%d",
            args.use_spectral,
            args.use_attention,
            args.alpha,
            args.wave,
            args.decomp_level,
        )
    else:
        logger.info(
            "DuoRec config | contrastive_weight=%.4f temperature=%.4f",
            args.contrastive_weight,
            args.contrastive_temperature,
        )

    model = MODEL_DICT[args.model_type.lower()](args=args)

    logger.info(model)
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    result_info = ""
    if args.do_eval:
        if args.load_model is None:
            logger.info("No model input!")
            exit(0)
        args.checkpoint_path = os.path.join(args.output_dir, args.load_model + ".pt")
        trainer.load(args.checkpoint_path)
        logger.info(f"Load model from {args.checkpoint_path} for test!")
        _, result_info = trainer.test(0)
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        training_start = time.time()
        for epoch in range(args.epochs):
            epoch_start = time.time()
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            elapsed = time.time() - epoch_start
            trainer.record_epoch_metrics(
                epoch=epoch,
                hr_at_10=float(scores[2]),
                ndcg_at_10=float(scores[3]),
                epoch_time_s=elapsed,
            )

            early_stopping(np.array([scores[3]]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        trainer.export_epoch_scores_csv()
        trainer.log_average_epoch_time()
        trainer.log_total_training_time(time.time() - training_start)
        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path, map_location=trainer.device))
        _, result_info = trainer.test(0)

    logger.info(args.train_name)
    logger.info(result_info)


if __name__ == "__main__":
    main()
