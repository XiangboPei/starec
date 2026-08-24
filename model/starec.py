import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import FeedForward, LayerNorm, MultiHeadAttention
from wavelet_fixed import WaveletPacketDecomposition


class StaRecModel(SequentialRecModel):
    def __init__(self, args):
        super(StaRecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = StaRecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        if all_sequence_output:
            return item_encoded_layers
        return item_encoded_layers[-1]

    def calculate_loss(self, input_ids, answers):
        """Full-vocabulary softmax cross-entropy (no pairwise BPR / sampled contrastive terms)."""
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)
        return loss


class StaRecEncoder(nn.Module):
    def __init__(self, args):
        super(StaRecEncoder, self).__init__()
        block = StaRecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class StaRecBlock(nn.Module):
    def __init__(self, args):
        super(StaRecBlock, self).__init__()
        self.layer = StaRecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        return self.feed_forward(layer_output)


class StaRecLayer(nn.Module):
    """Fixed wavelet packet branch and/or self-attention, fused with ``alpha`` when both are on.

    CLI: ``--use_spectral`` / ``--use_attention`` (default 1/1).
    Exactly one branch off yields the remaining branch alone; both off is invalid (see ``parse_args``).
    """

    def __init__(self, args):
        super(StaRecLayer, self).__init__()
        self.use_spectral = int(getattr(args, "use_spectral", 1)) == 1
        self.use_attention = int(getattr(args, "use_attention", 1)) == 1
        if not self.use_spectral and not self.use_attention:
            raise ValueError("StaRecLayer: at least one of use_spectral or use_attention must be on.")

        self.filter_layer = WaveletFrequencyLayer(args) if self.use_spectral else None
        self.attention_layer = MultiHeadAttention(args) if self.use_attention else None
        self.alpha = args.alpha

    def forward(self, input_tensor, attention_mask):
        if self.use_spectral and self.use_attention:
            dsp = self.filter_layer(input_tensor)
            gsp = self.attention_layer(input_tensor, attention_mask)
            return self.alpha * dsp + (1 - self.alpha) * gsp
        if self.use_spectral:
            return self.filter_layer(input_tensor)
        return self.attention_layer(input_tensor, attention_mask)


class WaveletFrequencyLayer(nn.Module):
    """Fixed wavelet packet path: learnable per-band scaling on detail coeffs, low-pass reconstruction only."""

    def __init__(self, args):
        super(WaveletFrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.decomp_level = int(getattr(args, "decomp_level", 1))

        self.wavelet = WaveletPacketDecomposition(
            J=self.decomp_level,
            wave=getattr(args, "wave", "sym4"),
        )

        self.wavelet_lowpass = nn.Parameter(torch.ones(1, 1, args.hidden_size))
        self.srt = nn.Parameter(torch.rand(args.hidden_size))
        self.level_scales = nn.ParameterList()
        filter_len = self.wavelet.filter_length
        cur_len = args.max_seq_length
        for _ in range(self.decomp_level):
            cur_len = (cur_len + filter_len - 2) // 2 + 1
            self.level_scales.append(nn.Parameter(torch.rand(cur_len, args.hidden_size)))

    def forward(self, input_tensor):
        x_permuted = input_tensor.permute(0, 2, 1)
        ca, cd = self.wavelet(x_permuted)
        for i in range(min(len(cd), len(self.level_scales))):
            scale = self.level_scales[i]
            cur = cd[i].permute(0, 2, 1)
            min_len = min(cur.shape[1], scale.shape[0])
            cur = cur[:, :min_len, :] * scale[:min_len, :] * (self.srt**2)
            cd[i] = cur.permute(0, 2, 1)

        sequence_emb_rec = self.wavelet.inverse((ca, cd)).permute(0, 2, 1)
        sequence_emb_rec = sequence_emb_rec * self.wavelet_lowpass

        hidden_states = self.out_dropout(sequence_emb_rec)
        return self.LayerNorm(hidden_states + input_tensor)
