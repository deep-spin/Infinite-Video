
# %%
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from utils.config import Config

# %%
import io

from models import VideoChat2_it_mistral
from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

import matplotlib.pyplot as plt

from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy

import json
from collections import OrderedDict

from tqdm import tqdm
import csv
import decord
import time
decord.bridge.set_bridge("torch")

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    # Make all arguments required
    parser.add_argument("--baseline", action="store_true", help="model.")
    parser.add_argument("--video-folder", required=True, help="Path to video folder.")
    parser.add_argument("--data_path", required=True, help="Path to question folder.")
    parser.add_argument('--output-dir', required=True, help="Directory to save the model results JSON.")
    
    # New arguments based on constants
    parser.add_argument("--max_int", type=int, required=True, help="Maximum integer value (e.g., 2048).")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples (e.g., 128).")
    parser.add_argument("--sticky", action="store_true", help="Set sticky flag (True or False).")
    parser.add_argument("--num_basis", type=int, required=True, help="Number of basis (e.g., 64 or 256).")
    parser.add_argument("--tau", type=float, required=True, help="Tau value (e.g., 0.75).")
    parser.add_argument("--alpha", type=float, required=True, help="alpha of weighted average between short and long term memory.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecated)."
    )
    return parser.parse_args()

def infer_egoschema(
        question, model, video, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        num_segments=8,
        video_path="",
    ):
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to("cuda:0")
    
    video_list = []
    with torch.no_grad():
        if system_q:
            video_emb, _ = model.encode_img(video, system + question)
        else:
            video_emb, _ = model.encode_img(video, system)
    video_list.append(video_emb)

    chat = EasyDict({
        "system": system,
        "roles": ("[INST]", "[/INST]"),
        "messages": [],
        "sep": ""
    })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> [/INST]"])
    
    prompt = question + question_prompt
    
    ask(prompt, chat)

    llm_message = answer(
        conv=chat, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt, print_res=print_res
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(llm_message)
    return llm_message

def infer_egoschema_inf(
        question, model, video, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        num_segments=1024,
        num_samples=8,
        video_path = ""
    ):
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to("cuda:0")
    video_list = torch.chunk(video, num_samples,  dim =1)
    video_embs = []
    new_video = True
    for video in video_list:
        with torch.no_grad():
            video_emb, _, = model.encode_img(video, system, new_video)
            video_embs.append(video_emb)
        new_video = False
    video_emb = torch.mean(torch.stack(video_embs), dim=0, keepdim=True).squeeze(0)  # Shape: [1, 32, 4096]
    video_list = [video_emb] 
    chat = EasyDict({
        "system": system,
        "roles": ("[INST]", "[/INST]"),
        "messages": [],
        "sep": ""
    })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> [/INST]"])
    
    if system_llm:
        prompt = system + question+ question_prompt
    else:
        prompt = question + question_prompt
    
    ask(prompt, chat)

    llm_message = answer(
        conv=chat, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt, print_res=print_res
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(llm_message)
    return llm_message


def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        #seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to("cuda:0"),
        torch.tensor([29871, 2]).to("cuda:0")]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
#     output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()
    
    


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs
    

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

def main(args):
    config_file = "configs/config_mistral.json"
    cfg = Config.from_file(config_file)
    cfg.model.vision_encoder.num_frames = 4
    cfg.model.sticky = args.sticky
    cfg.model.alpha = args.alpha
    cfg.model.tau= args.tau
    cfg.model.baseline = args.baseline
    cfg.model.num_basis = args.num_basis
    model = VideoChat2_it_mistral(config=cfg.model)

    # %%
    # add lora to run stage3 model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head"
        ]
    )
    model.mistral_model = get_peft_model(model.mistral_model, peft_config)

    # %%
    state_dict = torch.load("/mnt/data-poseidon/saul/Ask-Anything/video_chat2/VideoChat2_stage3_Mistral_7B/videochat2_mistral_7b_stage3.pth", "cpu")
                            
    if 'model' in state_dict.keys():
        msg = model.load_state_dict(state_dict['model'], strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    model = model.to(torch.device(cfg.device))
    model = model.eval()

    if args.baseline:
        output_dir = args.output_dir + f"/nframes_{args.max_int}_baseline_normal"  
    else:
        if args.sticky:
            sticky = "sticky"
        else:
            sticky= "uniform"
        output_dir = args.output_dir + f"/nframes_{args.max_int}_nchunks_{args.num_samples}_tau_{args.tau}_alpha_{args.alpha}_nbasis_{args.num_basis}_{sticky}_normal" 
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/preds.json"
        #  position embedding
    num_frame = args.max_int
    resolution = 224
    new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
    model.vision_encoder.encoder.pos_embed = new_pos_emb
    json_files = [f for f in os.listdir(args.data_path) if f.endswith('.json')]
    for file in json_files:
        file_path = os.path.join(args.data_path, file)
        with open(file_path, 'r') as json_file:
            movie_data = json.load(json_file)
            global_key = movie_data["info"]["video_path"]
            result_data = {}
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    result_data = json.load(f)
            if movie_data["info"]["video_path"] not in list(result_data.keys()):
                print(output_file)
                video_path = os.path.join(args.video_folder, movie_data["info"]["video_path"])
                if args.baseline:
                    video, _ = load_video(video_path, num_segments=args.max_int, return_msg=True)
                else:
                    video, _ = load_video(video_path, num_segments=int(args.max_int*args.num_samples), return_msg=True)
                global_value = []
                qa_data = movie_data["global"]
                for qa_key in qa_data:
                    start = time.time()
                    question = qa_key['question']
                    if args.baseline:
                        llm_message = infer_egoschema(
                            question, 
                            model,
                            video,
                            "You are able to understand the visual content that the user provides.Follow the instructions carefully and explain your answers.", 
                            question_prompt="", 
                            answer_prompt="",
                            return_prompt='',
                            system_q=False,
                            print_res=False,
                            system_llm=False,
                            num_segments=args.max_int,
                            video_path=args.video_folder,
                        )
                    else:
                        llm_message = infer_egoschema_inf(
                            question, 
                            model,
                            video,
                            "You are able to understand the visual content that the user provides.Follow the instructions carefully and explain your answers.", 
                            question_prompt="", 
                            answer_prompt="",
                            return_prompt='',
                            system_q=False,
                            print_res=False,
                            system_llm=False,
                            num_segments=int((args.max_int*args.num_samples)),
                            num_samples=args.num_samples,
                            video_path=args.video_folder,
                        )
                        
                    qa_key['pred'] = llm_message
                    global_value.append(qa_key)
                    
                    result_data[global_key]= global_value

                    if os.path.exists(output_file):
                        os.remove(output_file)  # Optional: write an empty list to reset the file

                    with open(output_file, 'a') as output_json_file:
                        json.dump(result_data, output_json_file, indent=4)
                    


if __name__ == '__main__':
    args = parse_args()
    main(args)