import tqdm
import torch
import numpy as np
import csv
import os
import time

from torch.optim import Adam
from metrics import recall_at_k, ndcg_k


class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger):
        super(Trainer, self).__init__()

        self.args = args
        self.logger = logger
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.epoch_scores = []
        self.epoch_times = []
        self.inference_times = []
        self.total_params = sum([p.nelement() for p in self.model.parameters()])

        self.logger.info(f"Total Parameters: {self.total_params}")

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        self.args.train_matrix = self.args.test_rating_matrix
        return self.iteration(epoch, self.test_dataloader, train=False)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        state_dict = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        # import pdb; pdb.set_trace()
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_full_sort_score(self, epoch, answers, pred_list):
        ks = (5, 10)
        hr = [recall_at_k(answers, pred_list, k) for k in ks]
        nd = [ndcg_k(answers, pred_list, k) for k in ks]
        post_fix = {
            "Epoch": epoch,
            "HR@5": "{:.4f}".format(hr[0]),
            "NDCG@5": "{:.4f}".format(nd[0]),
            "HR@10": "{:.4f}".format(hr[1]),
            "NDCG@10": "{:.4f}".format(nd[1]),
        }
        self.logger.info(post_fix)
        return [hr[0], nd[0], hr[1], nd[1]], str(post_fix)

    def record_epoch_metrics(self, epoch, hr_at_10, ndcg_at_10, epoch_time_s):
        self.epoch_scores.append(
            {
                "epoch": epoch,
                "HR@10": hr_at_10,
                "NDCG@10": ndcg_at_10,
                "s/epoch": epoch_time_s,
            }
        )
        self.epoch_times.append(epoch_time_s)

    def export_epoch_scores_csv(self, file_name=None):
        if file_name is None:
            file_name = self.args.train_name + "_epoch_scores.csv"
        csv_path = os.path.join(self.args.output_dir, file_name)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "HR@10", "NDCG@10"])
            for row in self.epoch_scores:
                writer.writerow([row["epoch"], row["HR@10"], row["NDCG@10"]])
        self.logger.info(f"Epoch scores exported to {csv_path}")

    def log_average_epoch_time(self):
        if not self.epoch_times:
            return
        avg_epoch_time = float(np.mean(self.epoch_times))
        self.logger.info(f"Average speed: {avg_epoch_time:.4f} s/epoch")

    def log_total_training_time(self, total_time):
        self.logger.info(f"Total training time: {total_time:.4f} s")

    def log_average_inference_time(self):
        if not self.inference_times:
            return
        avg_infer_time = float(np.mean(self.inference_times))
        self.logger.info(f"Average inference time: {avg_infer_time:.4f} s/eval")

    def iteration(self, epoch, dataloader, train=True):
        start_time = time.time()

        str_code = "train" if train else "test"
        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Mode_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        
        if train:
            self.model.train()
            rec_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)

                _, input_ids, answers, *extra_inputs = batch
                if extra_inputs:
                    loss = self.model.calculate_loss(
                        input_ids,
                        answers,
                        semantic_input_ids=extra_inputs[0],
                    )
                else:
                    loss = self.model.calculate_loss(input_ids, answers)
                    
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info(str(post_fix))

        else:
            self.model.eval()
            pred_list = None
            answer_list = None

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, answers = batch
                recommend_output = self.model.predict(input_ids, user_ids)
                recommend_output = recommend_output[:, -1, :]# 推荐的结果
                
                rating_pred = self.predict_full(recommend_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = -np.inf
                rating_pred[:, 0] = -np.inf

                # argpartition has O(n) complexity; argsort is O(n log n).
                # The minus sign "-" indicates a larger value.
                ind = np.argpartition(rating_pred, -10)[:, -10:]
                # Gather scores at the selected item indices for each top-k row.
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                # Sort the sub-tables in order of magnitude.
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                # retrieve the original subscript from index again
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            scores, info = self.get_full_sort_score(epoch, answer_list, pred_list)
            elapsed = time.time() - start_time
            self.inference_times.append(elapsed)
            self.logger.info(f"Inference time: {elapsed:.4f} s")
            self.log_average_inference_time()
            return scores, info
