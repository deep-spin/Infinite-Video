
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

import webvtt
import re

def clean_text(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return cleaned_text


def read_vtt_and_concatenate(file_path, tokenizer, max_len=4096):
    prev = ""
    subtitles = []
    for caption in webvtt.read(file_path):
        # Split the caption text into individual lines
        lines = caption.text.split('\n')
        for line in lines:
            # Clean the text and check for repetition
            line = clean_text(line)
            if prev != line and line:
                subtitles.append(line)
                prev = line

    # Join subtitles to check length
    full_text = ' '.join(subtitles)
    tokenized_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    # If the tokenized length is within the limit, return the full text
    if len(tokenized_ids) <= max_len:
        return full_text

    # Otherwise, we need to trim the text to fit within the limit
    # We will keep the first half and the last half
    half_len = max_len // 2
    start_text = ' '.join(subtitles[:half_len])
    end_text = ' '.join(subtitles[-half_len:])
    
    # Re-tokenize to ensure the total length is within the limit
    start_tokenized_ids = tokenizer(start_text, add_special_tokens=False).input_ids
    end_tokenized_ids = tokenizer(end_text, add_special_tokens=False).input_ids

    # Adjust the lengths to fit within the max_len
    while len(start_tokenized_ids) + len(end_tokenized_ids) > max_len:
        if len(start_tokenized_ids) > len(end_tokenized_ids):
            start_tokenized_ids.pop()
        else:
            end_tokenized_ids.pop(0)
    
    # Combine the adjusted parts
    adjusted_text = tokenizer.decode(start_tokenized_ids) + ' ... ' + tokenizer.decode(end_tokenized_ids)
    
    return adjusted_text

    
class MME_dataset(Dataset):
    def __init__(self, data_prefix, anno_path, num_segments=16, resolution=224, max_subtitle_len=4096):
        self.data_prefix = data_prefix
        with open(anno_path, 'r') as f:
            self.data_list = json.load(f)
            
        self.num_segments = num_segments
        self.max_subtitle_len = max_subtitle_len
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_frame(self, video_path, bound=None):
        video_path = os.path.join(video_path, str(self.num_segments))
        
        if os.path.exists(video_path):
            frame_list = [p for p in os.listdir(video_path)]
        else:
            raise Exception
            
        images_group = list()
        
        for frame_name in frame_list:
            img = Image.open(os.path.join(video_path, frame_name))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_video(self, video_path, bound=None):
        print("\n\n", video_path)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer = f"({answer}) {data['options'][ord(answer) - ord('A')][3:]}"
        for idx, c in enumerate(data['options']):
            cur_choice, cur_text = c[0], c[3:]
            question += f"({cur_choice}) {cur_text}\n"
        question = question.rstrip()
        return question, answer

    def __getitem__(self, idx):
        video_name = self.data_list[idx]['url'].split("watch?v=")[1]
        video_path = os.path.join(self.data_prefix, video_name + ".mp4")
        # We store the videos with only 16 or 32 frames for testing,
        # since directly reading the whold videos cost a lot of time.
        # You can also read the whole video via self.read_video(video_path)
        torch_imgs = self.read_video(video_path)
        duration_category = self.data_list[idx]['duration_category']
        qa_list = []
        for qa in self.data_list[idx]['questions']:
            qa_list.append(self.qa_template(qa))

        subtitle = ""
        try:
            subtitle_path = os.path.join(self.data_prefix, "subtitle", video_name + ".vtt")
            if os.path.exists(subtitle_path):
                subtitle = read_vtt_and_concatenate(subtitle_path, model.mistral_tokenizer, self.max_subtitle_len)
        except Exception:
            subtitle = ""
            print(f"Error for {subtitle_path}")
            
        return {
            'subtitle': subtitle,
            'video': torch_imgs, 
            'qa_list': qa_list,
            'duration_category': duration_category,
            "video_id": video_name
        }
    
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

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
def ask(text, conv):
    conv.messages.append([conv.roles[0], text])

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

def infer_mme(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        add_subtitle=False,
    ):
    assert system_q == False, "do not support system_q now"
    video = data_sample["video"]
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to("cuda:0")
    if not args.baseline:
        video_chunks = torch.chunk(video, args.num_samples,  dim =1)
    video_list = []
    with torch.no_grad():
        if system_q:
            raise NotImplementedError
        else:
            if args.baseline:
                video_emb, _ = model.encode_img(video, system)
            else:
                video_embs = 0
                new_video=True
                for vid in video_chunks:
                    video_embs, _ = model.encode_img(vid, system, new_video)
                    new_video=False
                    video_emb = i*video_emb/(i+1) + video_embs/(i+1)
    video_list.append(video_emb)

    pred_list = []
    gt_list = []
    for idx, qa in enumerate(data_sample['qa_list']):
        print(f"----------qa_{idx}---------", flush=True)
        chat = EasyDict({
            "system": system,
            "roles": ("[INST]", "[/INST]"),
            "messages": [],
            "sep": ""
        })
    
        if add_subtitle:
            if data_sample['subtitle'] != '':
                subtitle = f"This video's subtitles are listed below: {data_sample['subtitle']}"
                chat.messages.append([chat.roles[0], f"{subtitle}\n<Video><VideoHere></Video> [/INST]"])
            else:
                chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> [/INST]"])
        else:
            chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> [/INST]"])
    
        if system_llm:
            prompt = system + qa[0] + question_prompt
        else:
            prompt = qa[0] + question_prompt
        
        ask(prompt, chat)
    
        llm_message = answer(
            conv=chat, model=model, do_sample=False, 
            img_list=video_list, max_new_tokens=100, 
            answer_prompt=answer_prompt, print_res=print_res
        )[0]
        # remove potential explanation
        llm_message = return_prompt + llm_message.strip().split('\n')[0]
        pred_list.append(llm_message[1])
        gt_list.append(qa[1][1])
    return pred_list, gt_list

def main(args):
    config_file = "configs/config_mistral.json"
    cfg = Config.from_file(config_file)
    cfg.model.vision_encoder.num_frames = 4
    cfg.model.sticky = args.sticky
    cfg.model.alpha = args.alpha
    cfg.model.tau= args.tau
    cfg.model.num_basis = args.num_basis
    cfg.model.baseline = args.baseline
    global model
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
#  position embedding
    num_frame = args.max_int
    resolution = 224
    new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
    model.vision_encoder.encoder.pos_embed = new_pos_emb

    data_dir = args.video_folder
    anno_path =  args.data_path
    if args.baseline:
        dataset = MME_dataset(
            data_dir, 
            anno_path, 
            num_segments=num_frame, resolution=resolution
        )
    else:
        dataset = MME_dataset(
            data_dir, 
            anno_path, 
            num_segments=(num_frame*args.num_samples), resolution=resolution
        )
    with open(anno_path, 'r') as f:
        res_json_data = json.load(f)
    if args.sticky:
        sticky = "sticky"
    else:
        sticky= "uniform"
    if args.baseline:
        output_dir = args.output_dir + f"/nframes_{args.max_int}_baseline_normal"  
    else:
        output_dir = args.output_dir + f"/nframes_{args.max_int}_nchunks_{args.num_samples}_tau_{args.tau}_alpha_{args.alpha}_nbasis_{args.num_basis}_{sticky}_normal_mean"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/preds.json"

    res_list = []
    acc_dict = {}
    results ={}
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            acc_dict = json.load(f)
            res_list = acc_dict["res_list"]
    
    for idx, example in enumerate(tqdm(dataset)):
        if idx>=len(res_list):
            duration_category = example['duration_category']
            if duration_category not in acc_dict:
                i = idx 
            pred_list, gt_list = infer_mme(
                example, 
                "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
                question_prompt="\nOnly give the best option.",
                answer_prompt="Best option:(",
                return_prompt='(',
                system_q=False,
                print_res=False,
                system_llm=True,
                # add_subtitle=True, # Comment this line to add subtitles, we use the whole subtitles by default.
            )

            res_list.append({
                'pred': pred_list,
                'gt': gt_list
            })
            qa_idx=0
            for pred, gt in zip(pred_list, gt_list):
                res_json_data[idx]['questions'][qa_idx]['response'] = pred
                qa_idx += 1
            correct = 0
            total = 0
            total_cat = 0
            acc_dict[duration_category] = [0, 0, 0] # correct, total
            for idx1 in range(len(res_list)):
                key = res_list[idx1]  # Access the element at index idx1
                
                # Update total for all items
                total += len(key["pred"])
                
                # Update category-specific counts if idx1 >= i
                if idx1 >= i:
                    total_cat += len(key["pred"])
                    acc_dict[duration_category][1] = total_cat  # Total for the category

                    for pred, gt in zip(key["pred"], key["gt"]):
                        if pred == gt:
                            acc_dict[duration_category][0] += 1  # Correct for the category

                # Always check overall correctness
                for pred, gt in zip(key["pred"], key["gt"]):
                    if pred == gt:
                        correct += 1

            acc_dict[duration_category][2] += acc_dict[duration_category][0]/total_cat
            print(save_path, flush=True)
            print(f"Part  Acc: {acc_dict[duration_category][0] / acc_dict[duration_category][1] * 100 :.2f}%", flush=True)
            print(f"Total Acc: {correct / total * 100 :.2f}%", flush=True)
            print('-' * 50, duration_category, '-' * 50, flush=True)
            if os.path.exists(save_path):
                os.remove(save_path)  # Optional: write an empty list to reset the file
            with open(save_path, "w") as f:
                json.dump({
                    "acc_dict": acc_dict,
                    "res_list": res_list
                }, f)

            with open(f"{save_path}_full.json", "w") as f:
                json.dump(res_json_data, f)


if __name__ == '__main__':
    global args
    args = parse_args()
    main(args)
