"""DuoRec baseline with the repository's shared split and full-sort evaluator."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, TransformerEncoder


class DuoRecModel(SequentialRecModel):
    """SASRec encoder with DuoRec's default ``us_x`` contrastive objective."""

    def __init__(self, args):
        super().__init__(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = TransformerEncoder(args)
        self.temperature = float(args.contrastive_temperature)
        self.contrastive_weight = float(args.contrastive_weight)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        encoded_layers = self.item_encoder(
            sequence_emb,
            attention_mask,
            output_all_encoded_layers=True,
        )
        return encoded_layers if all_sequence_output else encoded_layers[-1]

    def calculate_loss(self, input_ids, answers, semantic_input_ids=None):
        sequence_output = self.forward(input_ids)[:, -1, :]
        logits = torch.matmul(sequence_output, self.item_embeddings.weight.transpose(0, 1))
        recommendation_loss = F.cross_entropy(logits, answers)

        if semantic_input_ids is None:
            raise ValueError("DuoRec training requires a same-target semantic positive sequence.")

        augmented_output = self.forward(input_ids)[:, -1, :]
        semantic_output = self.forward(semantic_input_ids)[:, -1, :]
        contrastive_logits, labels = self._info_nce(augmented_output, semantic_output)
        contrastive_loss = F.cross_entropy(contrastive_logits, labels)
        return recommendation_loss + self.contrastive_weight * contrastive_loss

    def _info_nce(self, first, second):
        batch_size = first.shape[0]
        representations = torch.cat((first, second), dim=0)
        similarities = torch.matmul(representations, representations.transpose(0, 1))
        similarities = similarities / self.temperature

        positive = torch.cat(
            (
                torch.diagonal(similarities, offset=batch_size),
                torch.diagonal(similarities, offset=-batch_size),
            )
        ).reshape(2 * batch_size, 1)

        mask = torch.ones(
            (2 * batch_size, 2 * batch_size),
            dtype=torch.bool,
            device=similarities.device,
        )
        mask.fill_diagonal_(False)
        indices = torch.arange(batch_size, device=similarities.device)
        mask[indices, batch_size + indices] = False
        mask[batch_size + indices, indices] = False
        negatives = similarities[mask].reshape(2 * batch_size, -1)

        contrastive_logits = torch.cat((positive, negatives), dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=similarities.device)
        return contrastive_logits, labels
