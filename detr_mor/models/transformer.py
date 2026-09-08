"""Standard (non-MoR) pre-norm transformer encoder and decoder for DETR."""

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    r"""
    Encoder for transformer of DETR.
    This has sequence of encoder layers.
    Each layer has the following modules:
        1. LayerNorm for Self Attention
        2. Self Attention
        3. LayerNorm for MLP
        4. MLP
    """

    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim,
                 dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Self Attention Module for all encoder layers
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_layers)
            ])

        # MLP Module for all encoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ])

        # Norm for Self Attention for all encoder layers
        self.attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Norm for MLP for all encoder layers
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Dropout for Self Attention for all encoder layers
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout for MLP for all encoder layers
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Norm for encoder output
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, spatial_position_embedding):
        r"""
        :param x: (B, num_tokens, d_model) flattened backbone features
        :param spatial_position_embedding: (num_tokens, d_model)
        :return: (encoder_output, attn_weights) where attn_weights is
            (num_layers, B, num_tokens, num_tokens)
        """
        out = x
        attn_weights = []
        for i in range(self.num_layers):
            # Norm, Self Attention, Dropout and Residual
            in_attn = self.attn_norms[i](out)
            # Add spatial position embedding to q,k for self attention
            q = in_attn + spatial_position_embedding
            k = in_attn + spatial_position_embedding
            out_attn, attn_weight = self.attns[i](
                query=q,
                key=k,
                value=in_attn
            )
            attn_weights.append(attn_weight)
            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, MLP, Dropout and Residual
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff

        # Output Normalization
        out = self.output_norm(out)
        return out, torch.stack(attn_weights)


class TransformerDecoder(nn.Module):
    r"""
    Decoder for transformer of DETR.
    This has sequence of decoder layers.
    Each layer has the following modules:
        1. LayerNorm for Self Attention
        2. Self Attention
        3. LayerNorm for Cross Attention on Encoder Outputs
        4. Cross Attention
        5. LayerNorm for MLP
        6. MLP
    """

    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim,
                 dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Self Attention module for all decoder layers
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_layers)
            ])

        # Cross Attention Module for all decoder layers
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_layers)
            ])

        # MLP Module for all decoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ])

        # Norm for Self Attention Module for all decoder layers
        self.attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Norm for Cross Attention Module for all decoder layers
        self.cross_attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Norm for MLP Module for all decoder layers
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Dropout for Attention Module for all decoder layers
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout for Cross Attention Module for all decoder layers
        self.cross_attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout for MLP Module for all decoder layers
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Shared Output norm for all decoder outputs
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query_objects, encoder_output,
                query_embedding, spatial_position_embedding):
        r"""
        :param query_objects: (B, num_queries, d_model), zeros at the first layer
        :param encoder_output: (B, num_tokens, d_model)
        :param query_embedding: (B, num_queries, d_model) learned object queries
        :param spatial_position_embedding: (num_tokens, d_model)
        :return: (outputs, cross_attn_weights) where outputs is
            (num_layers, B, num_queries, d_model) - one entry per decoder layer,
            for deep supervision.
        """
        out = query_objects
        decoder_outputs = []
        decoder_cross_attn_weights = []
        for i in range(self.num_layers):
            # Norm, Self Attention, Dropout and Residual
            in_attn = self.attn_norms[i](out)
            q = in_attn + query_embedding
            k = in_attn + query_embedding
            out_attn, _ = self.attns[i](
                query=q,
                key=k,
                value=in_attn
            )
            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, Cross Attention, Dropout and Residual
            in_attn = self.cross_attn_norms[i](out)
            q = in_attn + query_embedding
            k = encoder_output + spatial_position_embedding
            out_attn, decoder_cross_attn = self.cross_attns[i](
                query=q,
                key=k,
                value=encoder_output
            )
            decoder_cross_attn_weights.append(decoder_cross_attn)
            out_attn = self.cross_attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, MLP, Dropout and Residual
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff
            decoder_outputs.append(self.output_norm(out))

        output = torch.stack(decoder_outputs)
        return output, torch.stack(decoder_cross_attn_weights)
