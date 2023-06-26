#!/usr/bin/env python

import os, sys
from typing import Dict

import torch
from safetensors.torch import load_file

__all__ = ['weight_loader_fn_dict']

def load_weights_clip(load_path: str) -> Dict[str, torch.Tensor]:
    clip_model = torch.jit.load(load_path, map_location='cpu')
    clip_model = clip_model.visual
    src_state_dict = clip_model.state_dict()
    src_state_dict = dict((k, v.float()) for k, v in src_state_dict.items())

    dst_state_dict = {}
    
    dst_state_dict['cls_token'] = src_state_dict['class_embedding']
    dst_state_dict['pos_embed'] = src_state_dict['positional_embedding']
    dst_state_dict['patch_embed.proj.weight'] = src_state_dict['conv1.weight'].flatten(1)
    dst_state_dict['patch_embed.proj.bias'] = torch.zeros([src_state_dict['conv1.weight'].size(0)])
    
    dst_state_dict['ln_pre.weight'] = src_state_dict['ln_pre.weight']
    dst_state_dict['ln_pre.bias'] = src_state_dict['ln_pre.bias']

    block_idx = 0
    while True:
        src_prefix = 'transformer.resblocks.%d.' % block_idx
        dst_prefix = 'blocks.%d.' % block_idx

        src_block_state_dict = dict((k[len(src_prefix):], v) for k, v in src_state_dict.items() if k.startswith(src_prefix))
        if len(src_block_state_dict) == 0:
            break

        dst_block_state_dict = {}
        feat_dim = src_block_state_dict['ln_1.weight'].size(0)

        for i, dst_name in enumerate(('q', 'k', 'v')):
            dst_block_state_dict['attn.%s_proj.weight' % dst_name] = src_block_state_dict['attn.in_proj_weight'][feat_dim * i: feat_dim * (i + 1)]
            dst_block_state_dict['attn.%s_proj.bias' % dst_name] = src_block_state_dict['attn.in_proj_bias'][feat_dim * i: feat_dim * (i + 1)]
        
        dst_block_state_dict['attn.out_proj.weight'] = src_block_state_dict['attn.out_proj.weight']
        dst_block_state_dict['attn.out_proj.bias'] = src_block_state_dict['attn.out_proj.bias']

        dst_block_state_dict['mlp.fc1.weight'] = src_block_state_dict['mlp.c_fc.weight']
        dst_block_state_dict['mlp.fc1.bias'] = src_block_state_dict['mlp.c_fc.bias']
        dst_block_state_dict['mlp.fc2.weight'] = src_block_state_dict['mlp.c_proj.weight']
        dst_block_state_dict['mlp.fc2.bias'] = src_block_state_dict['mlp.c_proj.bias']

        dst_block_state_dict['norm1.weight'] = src_block_state_dict['ln_1.weight']
        dst_block_state_dict['norm1.bias'] = src_block_state_dict['ln_1.bias']
        dst_block_state_dict['norm2.weight'] = src_block_state_dict['ln_2.weight']
        dst_block_state_dict['norm2.bias'] = src_block_state_dict['ln_2.bias']

        dst_state_dict.update(dict((dst_prefix + k, v) for k, v in dst_block_state_dict.items()))
        block_idx += 1

    return dst_state_dict

def load_weights_custom_clip(load_path):
    state_dict = load_file(load_path)
    
    def rename(name):
        name = name.removeprefix('student.vision_model.')
        if 'class_embedding' in name:
            return 'cls_token'
        elif 'patch_embedding' in name:
            return name.replace('embeddings.patch_embedding', 'patch_embed.proj')
        elif 'position_embedding' in name:
            return 'pos_embed'
        elif 'encoder' in name:
            return (
                name
                .replace('encoder.layers', 'blocks')
                .replace('layer_norm', 'norm')
                .replace('self_attn', 'attn')
            )
        elif 'post_layernorm' in name:
            return name.replace('post_layernorm', 'ln_post')
        else:
            print(name)
            raise Exception()

    new_state_dict = {rename(k): v for k, v in state_dict.items() if k.startswith('student.vision_model') and 'position_ids' not in k}
    return new_state_dict


weight_loader_fn_dict = {
    'clip': load_weights_clip,
    'custom_clip': load_weights_custom_clip
}
