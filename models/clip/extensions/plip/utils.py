import torch
from transformers import CLIPModel
import os
import json
import argparse
from collections import OrderedDict


def convert_state_dict_from_transformers(openclip_state_dict, hf_state_dict, verbose=True):

    converted_dict = OrderedDict()
    
    for k in openclip_state_dict.keys():
        converted_dict[k] = openclip_state_dict[k]
    
    mapping = {
        "visual.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
        "visual.class_embedding": "vision_model.embeddings.class_embedding",
        "visual.positional_embedding": "vision_model.embeddings.position_embedding.weight",
        "visual.ln_pre.weight": "vision_model.pre_layrnorm.weight",
        "visual.ln_pre.bias": "vision_model.pre_layrnorm.bias",
        "visual.ln_post.weight": "vision_model.post_layernorm.weight",
        "visual.ln_post.bias": "vision_model.post_layernorm.bias",
        "visual.proj": "visual_projection.weight",  
        
        "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
        "positional_embedding": "text_model.embeddings.position_embedding.weight",
        "ln_final.weight": "text_model.final_layer_norm.weight",
        "ln_final.bias": "text_model.final_layer_norm.bias",
        "text_projection": "text_projection.weight",  

        "logit_scale": "logit_scale",
    }
    
  
    for i in range(12):  

        mapping[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"] = None 
        mapping[f"visual.transformer.resblocks.{i}.attn.in_proj_bias"] = None  
        mapping[f"visual.transformer.resblocks.{i}.attn.out_proj.weight"] = f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        mapping[f"visual.transformer.resblocks.{i}.attn.out_proj.bias"] = f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        
        mapping[f"visual.transformer.resblocks.{i}.ln_1.weight"] = f"vision_model.encoder.layers.{i}.layer_norm1.weight"
        mapping[f"visual.transformer.resblocks.{i}.ln_1.bias"] = f"vision_model.encoder.layers.{i}.layer_norm1.bias"
        mapping[f"visual.transformer.resblocks.{i}.ln_2.weight"] = f"vision_model.encoder.layers.{i}.layer_norm2.weight"
        mapping[f"visual.transformer.resblocks.{i}.ln_2.bias"] = f"vision_model.encoder.layers.{i}.layer_norm2.bias"
        
        mapping[f"visual.transformer.resblocks.{i}.mlp.c_fc.weight"] = f"vision_model.encoder.layers.{i}.mlp.fc1.weight"
        mapping[f"visual.transformer.resblocks.{i}.mlp.c_fc.bias"] = f"vision_model.encoder.layers.{i}.mlp.fc1.bias"
        mapping[f"visual.transformer.resblocks.{i}.mlp.c_proj.weight"] = f"vision_model.encoder.layers.{i}.mlp.fc2.weight"
        mapping[f"visual.transformer.resblocks.{i}.mlp.c_proj.bias"] = f"vision_model.encoder.layers.{i}.mlp.fc2.bias"
    
    for i in range(12):  
        mapping[f"transformer.resblocks.{i}.attn.in_proj_weight"] = None  
        mapping[f"transformer.resblocks.{i}.attn.in_proj_bias"] = None 
        mapping[f"transformer.resblocks.{i}.attn.out_proj.weight"] = f"text_model.encoder.layers.{i}.self_attn.out_proj.weight"
        mapping[f"transformer.resblocks.{i}.attn.out_proj.bias"] = f"text_model.encoder.layers.{i}.self_attn.out_proj.bias"
        
        mapping[f"transformer.resblocks.{i}.ln_1.weight"] = f"text_model.encoder.layers.{i}.layer_norm1.weight"
        mapping[f"transformer.resblocks.{i}.ln_1.bias"] = f"text_model.encoder.layers.{i}.layer_norm1.bias"
        mapping[f"transformer.resblocks.{i}.ln_2.weight"] = f"text_model.encoder.layers.{i}.layer_norm2.weight"
        mapping[f"transformer.resblocks.{i}.ln_2.bias"] = f"text_model.encoder.layers.{i}.layer_norm2.bias"
        
        mapping[f"transformer.resblocks.{i}.mlp.c_fc.weight"] = f"text_model.encoder.layers.{i}.mlp.fc1.weight"
        mapping[f"transformer.resblocks.{i}.mlp.c_fc.bias"] = f"text_model.encoder.layers.{i}.mlp.fc1.bias"
        mapping[f"transformer.resblocks.{i}.mlp.c_proj.weight"] = f"text_model.encoder.layers.{i}.mlp.fc2.weight"
        mapping[f"transformer.resblocks.{i}.mlp.c_proj.bias"] = f"text_model.encoder.layers.{i}.mlp.fc2.bias"
    
    for openclip_key, hf_key in mapping.items():

        if hf_key is None:
            continue

        if hf_key in hf_state_dict:
            if openclip_key == "visual.proj" or openclip_key == "text_projection":
                converted_dict[openclip_key] = hf_state_dict[hf_key].t()
            else:
                if openclip_key in openclip_state_dict:
                    if openclip_state_dict[openclip_key].shape == hf_state_dict[hf_key].shape:
                        converted_dict[openclip_key] = hf_state_dict[hf_key]
                    else:
                        if verbose:
                            print(f"unmatched shape: {openclip_key} vs {hf_key}, " 
                                  f"OpenCLIP shape: {openclip_state_dict[openclip_key].shape}, "
                                  f"HF shape: {hf_state_dict[hf_key].shape}")
                else:
                    if verbose:
                        print(f"cannot find key in OpenCLIP statedic: {openclip_key}")
        else:
            if verbose:
                print(f"cannot find key in HF statedic: {hf_key}")
    

    for i in range(12):
        try:
            if all(k in hf_state_dict for k in [
                f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight",
                f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight",
                f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
            ]):
                vision_in_proj_weight = torch.cat([
                    hf_state_dict[f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"],
                    hf_state_dict[f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"],
                    hf_state_dict[f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"]
                ])
                converted_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"] = vision_in_proj_weight
                
                vision_in_proj_bias = torch.cat([
                    hf_state_dict[f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"],
                    hf_state_dict[f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"],
                    hf_state_dict[f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"]
                ])
                converted_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_bias"] = vision_in_proj_bias
        except Exception as e:
            if verbose:
                print(f"Error in vision model layer {i} : {e}")
        
        try:
            if all(k in hf_state_dict for k in [
                f"text_model.encoder.layers.{i}.self_attn.q_proj.weight",
                f"text_model.encoder.layers.{i}.self_attn.k_proj.weight",
                f"text_model.encoder.layers.{i}.self_attn.v_proj.weight"
            ]):
                text_in_proj_weight = torch.cat([
                    hf_state_dict[f"text_model.encoder.layers.{i}.self_attn.q_proj.weight"],
                    hf_state_dict[f"text_model.encoder.layers.{i}.self_attn.k_proj.weight"],
                    hf_state_dict[f"text_model.encoder.layers.{i}.self_attn.v_proj.weight"]
                ])
                converted_dict[f"transformer.resblocks.{i}.attn.in_proj_weight"] = text_in_proj_weight

                text_in_proj_bias = torch.cat([
                    hf_state_dict[f"text_model.encoder.layers.{i}.self_attn.q_proj.bias"],
                    hf_state_dict[f"text_model.encoder.layers.{i}.self_attn.k_proj.bias"],
                    hf_state_dict[f"text_model.encoder.layers.{i}.self_attn.v_proj.bias"]
                ])
                converted_dict[f"transformer.resblocks.{i}.attn.in_proj_bias"] = text_in_proj_bias
        except Exception as e:
            if verbose:
                print(f"Error in text model layer {i}: {e}")
    
    missing_keys = []
    for k in openclip_state_dict.keys():
        if k not in converted_dict or converted_dict[k] is None:
            missing_keys.append(k)
    
    if missing_keys and verbose:
        print(f"wanrning: {len(missing_keys)} keys not mapped:")
        for k in missing_keys[:10]:  
            print(f"  - {k}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} other keys")
    
    return converted_dict





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", 
                        )
    parser.add_argument("--output_dir", type=str, default="./converted_clip", 
                        )
    parser.add_argument("--test", action="store_true", 
                        )
    parser.add_argument("--image_path", type=str, 
                        )
    parser.add_argument("--text", type=str, default="a photo of a cat", 
                        )
    parser.add_argument("--verbose", action="store_true", default=True,
                        )
    
    args = parser.parse_args()
    
    convert_transformers_clip_to_openclip(
        args.model_name, 
        args.output_dir,
        verbose=args.verbose
    )
    