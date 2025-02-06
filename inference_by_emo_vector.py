# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import os
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params_esd_fine_tuning as params
from model import GradTTS
from text import phone_text_to_sequence
from text.symbols import symbols
from utils import intersperse

from decimal import Decimal
import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

import pdb

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

# HIFIGAN_CONFIG = 'checkpts/config.json'
# HIFIGAN_CHECKPT = 'checkpts/g_02000000'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, 
                        default='resources/filelists/esd_22050/eva.txt',
                        help='path to a file with texts to synthesize')
    parser.add_argument('--with_emo', type=str, default='with_emo', help='')
    parser.add_argument('--emo1', type=str, default=None, help='select a emo expert model of A')
    parser.add_argument('--emo2', type=str, default=None, help='select a emo expert model of B')
    parser.add_argument('--int1', type=float, default=None, help='select a emo expert model of B')
    parser.add_argument('--int2', type=float, default=None, help='select a emo expert model of B')
    parser.add_argument('--alpha', type=float, default=None, help='precent for emo A')
    # parser.add_argument('--gamma', type=float, default=0.5, help='precent for emo B')

    # parser.add_argument('--pre_ckpt', type=str, 
    #                     default='logs_new/ljs_pre_no_emo_new/grad_ljs_1000.pt', 
    #                     help='path to a checkpoint of Grad-TTS')
    parser.add_argument('--resources_path', type=str,
                        default='resources/filelists/esd_22050/')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps of reverse diffusion')
    parser.add_argument('--speaker_id', type=str, default='13', help='speaker id for multispeaker model')
    parser.add_argument('--result_dir', type=str, default='out_new_new', help='out dir')
    parser.add_argument("--use_finer_emo_feature_to_train", type=bool, default=True)
    args = parser.parse_args()
    
    assert (args.with_emo == 'with_emo' and args.use_finer_emo_feature_to_train) or (args.with_emo == 'no_emo' and not args.use_finer_emo_feature_to_train)

    pre_ckpt = f'logs_new/ljs_pre_{args.with_emo}_new/grad_ljs_1000.pt'
    resources_path = args.resources_path

    print('Initializing Grad-TTS...')
    # spks = params.n_spks
    generator = GradTTS(len(symbols)+1, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale, 
                        params.x_vector_dim, params.emo_dem_dim, args)
    
    pre_ckpt = torch.load(pre_ckpt, map_location=lambda loc, storage: loc)['model']
    emo_vec_A, emo_vec_B = None, None
    if args.emo1 is not None:
        emo_vec_A = torch.load(f'emotion_vector/{args.speaker_id}/{args.with_emo}/emotion_vector_{args.emo1}.pt', map_location=lambda loc, storage: loc)
    if args.emo2 is not None:
        emo_vec_B = torch.load(f'emotion_vector/{args.speaker_id}/{args.with_emo}/emotion_vector_{args.emo2}.pt', map_location=lambda loc, storage: loc)
    
    
    # final_ckpt = {}
    # for key in pre_ckpt.keys():
    #     # pdb.set_trace()
    #     if emo_vec_A is not None and emo_vec_B is None:
    #         final_ckpt[key] = pre_ckpt[key] + emo_vec_A[key]
    #     if emo_vec_A is not None and emo_vec_B is not None: 
    #         if args.alpha is None:
    #             final_ckpt[key] = pre_ckpt[key] + args.int1 * emo_vec_A[key] + args.int2 * emo_vec_B[key]
    #         else:
    #             final_ckpt[key] = pre_ckpt[key] + args.alpha * emo_vec_A[key] + float(Decimal(str(1)) - Decimal(str(args.alpha))) * emo_vec_B[key]

    # generator.load_state_dict(final_ckpt)
    # _ = generator.cuda().eval()
    # print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    with open(args.file, 'r', encoding='utf-8') as f:
        infos = [line.strip() for line in f.readlines() if line.split('|')[1] == args.speaker_id]
    # pdb.set_trace()
    # filter out others speaker

    if args.emo1 is not None and args.emo2 is None:
        save_dir = os.path.join(args.result_dir, args.speaker_id, f'primary_{args.with_emo}_{args.emo1}')
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        final_ckpt = {}
        for key in pre_ckpt.keys():
            # pdb.set_trace()
            final_ckpt[key] = pre_ckpt[key] + emo_vec_A[key]

        generator.load_state_dict(final_ckpt)
        _ = generator.cuda().eval()
        print(f'Number of parameters: {generator.nparams}')

        infos = [i for i in infos if i.split('|')[1] == args.speaker_id and i.split('|')[-1] == args.emo1]
        with torch.no_grad():
            for i, info in enumerate(infos):
                basename, speaker, phn, text, _ = info.split('|')

                print(f'Synthesizing {i} text...', end=' ')
                n_phns = len(phn.split())
                x = torch.LongTensor(phone_text_to_sequence(phn)).cuda()[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                
                spk_emb_path = os.path.join(resources_path, 'spk_emb', speaker, f'{basename}.npy')
                spk_emb = torch.FloatTensor(np.load(spk_emb_path)).cuda()[None]
                # pdb.set_trace()
                emo_finer_path = os.path.join(resources_path, 'emotion_demension_finer', speaker, f'{basename}.npy')
                emo_finer = torch.FloatTensor(np.load(emo_finer_path))
                emo_finer = torch.nn.functional.interpolate(emo_finer.unsqueeze(0).permute(0,2,1), size=(n_phns), mode='linear')
                emo_finer = emo_finer.permute(0,2,1).cuda()
                print(emo_finer.shape)
                t = dt.datetime.now()

                y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                    stoc=False, spk_emb=spk_emb, emo_finer=emo_finer, length_scale=1)

                t = (dt.datetime.now() - t).total_seconds()
                print(f'Grad-TTS RTF: {t * params.sample_rate / (y_dec.shape[-1] * 256)}')

                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                
                write(f'{save_dir}/{basename}.wav', params.sample_rate, audio)

        print('Done. Check out `out` folder for samples.')

    if args.emo1 is not None and args.emo2 is not None:

        if args.alpha is None:
            save_dir = os.path.join(args.result_dir, args.speaker_id, f'emomixure_{args.with_emo}_new', f'{args.int1}_{args.emo1}_{args.int2}_{args.emo2}/')
        else:
            save_dir = os.path.join(args.result_dir, args.speaker_id, f'emomixure_{args.with_emo}_new', f'{args.alpha}_{args.emo1}_{float(Decimal(str(1)) - Decimal(str(args.alpha)))}_{args.emo2}/')
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        final_ckpt = {}
        for key in pre_ckpt.keys():
            # pdb.set_trace()
            if args.alpha is None:
                final_ckpt[key] = pre_ckpt[key] + args.int1 * emo_vec_A[key] + args.int2 * emo_vec_B[key]
            else:
                final_ckpt[key] = pre_ckpt[key] + args.alpha * emo_vec_A[key] + float(Decimal(str(1)) - Decimal(str(args.alpha))) * emo_vec_B[key]

        generator.load_state_dict(final_ckpt)
        _ = generator.cuda().eval()
        print(f'Number of parameters: {generator.nparams}')

        neu_infos = [i for i in infos if i.split('|')[1] == args.speaker_id and i.split('|')[-1] == 'neu']
        neu_infos = sorted(neu_infos)

        emo1_infos, emo2_infos = None, None
        if args.emo1 is not None:
            emo1_infos = [i for i in infos if i.split('|')[1] == args.speaker_id and i.split('|')[-1] == args.emo1]
            emo1_infos = sorted(emo1_infos)
        if args.emo2 is not None:
            emo2_infos = [i for i in infos if i.split('|')[1] == args.speaker_id and i.split('|')[-1] == args.emo2]
            emo2_infos = sorted(emo2_infos)

        # pdb.set_trace()
        with torch.no_grad():
            for i, (neu_info, emo1_info, emo2_info) in enumerate(zip(neu_infos, emo1_infos, emo2_infos)):
                basename, speaker, phn, text, _ = neu_info.split('|')
                emo1_basename, _, _, _, _ = emo1_info.split('|')
                emo2_basename, _, _, _, _ = emo2_info.split('|')

                n_phns = len(phn.split())

                print(f'Synthesizing {i} text...', end=' ')
                x = torch.LongTensor(phone_text_to_sequence(phn)).cuda()[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                
                spk_emb_path = os.path.join(resources_path, 'spk_emb', speaker, f'{basename}.npy')
                spk_emb = torch.FloatTensor(np.load(spk_emb_path)).cuda()[None]

                if args.use_finer_emo_feature_to_train:
                    emo1_finer_path = os.path.join(resources_path, 'emotion_demension_finer', speaker, f'{emo1_basename}.npy')
                    emo1_finer = torch.FloatTensor(np.load(emo1_finer_path))
                    emo1_finer = torch.nn.functional.interpolate(emo1_finer.unsqueeze(0).permute(0,2,1), size=(n_phns), mode='linear')
                    emo1_finer = emo1_finer.permute(0,2,1).cuda()
                    
                    emo2_finer_path = os.path.join(resources_path, 'emotion_demension_finer', speaker, f'{emo2_basename}.npy')
                    emo2_finer = torch.FloatTensor(np.load(emo2_finer_path))
                    emo2_finer = torch.nn.functional.interpolate(emo2_finer.unsqueeze(0).permute(0,2,1), size=(n_phns), mode='linear')
                    emo2_finer = emo2_finer.permute(0,2,1).cuda()

                    # pdb.set_trace()
                    emo_finer = args.alpha * emo1_finer + float(Decimal(str(1)) - Decimal(str(args.alpha))) * emo2_finer
                    print(emo_finer.shape)
                else:
                    emo_finer = None

                t = dt.datetime.now()

                y_enc, y_dec, attn = generator(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                    stoc=False, spk_emb=spk_emb, emo_finer=emo_finer, length_scale=1)

                t = (dt.datetime.now() - t).total_seconds()
                print(f'Grad-TTS RTF: {t * params.sample_rate / (y_dec.shape[-1] * 256)}')

                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                
                write(f'{save_dir}/{basename}.wav', params.sample_rate, audio)

        print('Done. Check out `out` folder for samples.')
