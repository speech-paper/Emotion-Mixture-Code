# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch
import torch.nn as nn
from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility
from model.utils import pad
import pdb

class GradTTS(BaseModule):
    def __init__(self, n_vocab, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale, x_vector_dim, emo_dem_dim, args):
        super(GradTTS, self).__init__()
        self.n_vocab = n_vocab
        # self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.use_finer = args.use_finer_emo_feature_to_train
        print(f'use-finer: {self.use_finer}')

        # if n_spks > 1:
        #     self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.spk_proj = nn.Sequential(nn.Linear(x_vector_dim, spk_emb_dim),
                                      nn.ReLU(),
                                      nn.Linear(spk_emb_dim, spk_emb_dim))
        if self.use_finer:
            self.emo_encoder = nn.Sequential(nn.Linear(emo_dem_dim, n_enc_channels),
                                      nn.ReLU(),
                                      nn.Linear(n_enc_channels, n_enc_channels))
                               
        # n_vocab=149 , n_feats=80, n_enc_channels=192, filter_channels=768, 
        # filter_channels_dp=256, n_heads=2, n_enc_layers=6, enc_kernel=3, enc_dropout
        # window_size=4
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size, use_finer=self.use_finer)

        self.decoder = Diffusion(n_feats, dec_dim, spk_emb_dim, beta_min, beta_max, pe_scale)

        self.mse_loss = nn.MSELoss()

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, spk_emb=None, 
                emo_finer=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        # if self.n_spks > 1:
        #     # Get speaker embedding
        #     spk = self.spk_emb(spk)

        spk = self.spk_proj(spk_emb)

        # add emo finer
        # print(emo_finer.shape)
        if self.use_finer:
            emo_finer = self.emo_encoder(emo_finer).permute(0,2,1)
        else:
            # print('False')
            emo_finer = torch.zeros([x.shape[0], x.shape[1], self.n_enc_channels]).to(x.device).permute(0,2,1)
            # print(emo_finer)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk, emo_finer)

        w = torch.exp(logw) * x_mask          # 
        w_ceil = torch.ceil(w) * length_scale  
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long() # clamp_min，设置下界，<1则=1
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length) # 判断是否为4的倍数

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2) # [B,1,text_len,1] [B,1,1,mel_len]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        # mu_y, _ = self.length_regulator(mu_x.permute(0, 2, 1), w_ceil.squeeze(1), y_max_length_)
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, d_targets, d_lengths, spk=None, spk_emb=None, 
                     emo_finer=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            d_targets (torch.Tensor): batch of corresponding phoneme duration target.
            d_lengths (torch.Tensor): lengths of phoneme duration in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        # pdb.set_trace()
        x, x_lengths, y, y_lengths, d_targets, d_lengths = self.relocate_input([x, x_lengths, y, y_lengths, d_targets, d_lengths])

        # if self.n_spks > 1:
        #     # Get speaker embedding
        #     spk = self.spk_emb(spk)
        
        # projection x_vector_dim 256 to 64
        spk = self.spk_proj(spk_emb)

        # add emo finer
        if self.use_finer:
            emo_finer = self.emo_encoder(emo_finer).permute(0,2,1)
        else:
            emo_finer = torch.zeros([x.shape[0], x.shape[1], self.n_enc_channels]).to(x.device).permute(0,2,1)
        # print(emo_finer.shape)
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`# 这里说的编码器输出实际上就是transformer encoder，只是TextEncoder的一部分
        mu_x, log_duration_prediction, x_mask = self.encoder(x, x_lengths, spk, emo_finer=emo_finer) #[B,80,L] [B,1,L] [B,1,L]

        log_duration_targets = torch.log(1e-8 + d_targets.float())
        log_duration_targets.requires_grad = False
        src_mask = x_mask.squeeze(1).bool()  # 创建掩码布尔张量
        log_duration_predictions = log_duration_prediction.squeeze(1)
        log_duration_targets = log_duration_targets.masked_select(src_mask)
        log_duration_predictions = log_duration_predictions.masked_select(src_mask)
        dur_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        
        y_max_length = y.shape[-1] 
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2) # [B,1,x_max_len,y_max_len]
        attn = generate_path(d_targets, attn_mask.squeeze(1)).unsqueeze(1)


        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))       
        mu_y = mu_y.transpose(1, 2) # [B, 80, 172]
        # pdb.set_trace()

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)

            if torch.sum(out_offset) == 0: # 每个片段都小于out_size
                print(y_lengths)
                y_cut   =  torch.zeros(y.shape[0], self.n_feats, max(y_lengths).item(), dtype=y.dtype, device=y.device)# [B,80,172]
                mu_y_cut = torch.zeros(y.shape[0], self.n_feats, max(y_lengths).item(), dtype=y.dtype, device=y.device)# [B,80,172]
            else:
                y_cut   =  torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)# [B,80,172]
                mu_y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)# [B,80,172]
            y_cut_lengths = []
            for i, (y_, mu_y_, out_offset_) in enumerate(zip(y, mu_y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                # y_cut_length = min(y_cut_length, y_.shape[-1])
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                mu_y_cut[i, :, :y_cut_length] = mu_y_[:, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask) # [B,1,172]
            
            y = y_cut
            mu_y = mu_y_cut
            y_mask = y_cut_mask

        assert y.shape[-1] == y_mask.shape[-1], f'{y.shape[-1]}, {y_mask.shape[-1]}'
        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return dur_loss, prior_loss, diff_loss
    
    def extract_lev(self, emo_finer):
        return self.emo_encoder(emo_finer)
    
    def get_ling_info(self, x, x_lengths):
        return self.encoder.get_ling_infos(x, x_lengths)
    