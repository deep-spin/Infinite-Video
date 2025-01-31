
# %%
import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
from utils.config import Config

# %%
import io

from models import VideoChat2_it_hd_mistral
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
        data_sample, model, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        num_segments=8,
        video_path = ""
    
    ):
    vid_path = os.path.join(video_path, data_sample['video'])
    video, _ = load_video(vid_path, num_segments=num_segments, return_msg=True)
    T_, C, H, W = video.shape
    video = video.reshape(1, T_, C, H, W).to("cuda:0")
    
    video_list = []
    with torch.no_grad():
        if system_q:
            video_emb, _, _ = model.encode_img(video, system + data_sample['question'])
        else:
            video_emb, _, _ = model.encode_img(video, system)
    video_list.append(video_emb[0])

    chat = EasyDict({
        "system": system,
        "roles": ("[INST]", "[/INST]"),
        "messages": [],
        "sep": ""
    })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> [/INST]"])
    
    if system_llm:
        prompt = system + data_sample['QA'][0]['q'] + question_prompt
    else:
        prompt = data_sample['QA'][0]['q'] + question_prompt
    
    ask(prompt, chat)

    llm_message = answer(
        conv=chat, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt, print_res=print_res
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(llm_message)
    print(f"GT: {data_sample['QA'][0]['a']}")
    return llm_message

def infer_egoschema_inf(
        data_sample, model, system="", 
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
    vid_path = os.path.join(video_path, data_sample['video'])
    video, _ = load_video(vid_path, num_segments=num_segments, return_msg=True)
    T_, C, H, W = video.shape
    video = video.reshape(1, T_, C, H, W).to("cuda:0")
    video_list = torch.chunk(video, num_samples,  dim =1)
    video_embs = []
    new_video = True
    for video in video_list:
        with torch.no_grad():
            if system_q:
                video_emb, _, _ = model.encode_img(video, system + data_sample['question'], new_video)
            else:
                video_emb, _, _ = model.encode_img(video, system, new_video)
        video_embs.append(video_emb[0])
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
        prompt = system + data_sample['QA'][0]['q'] + question_prompt
    else:
        prompt = data_sample['QA'][0]['q'] + question_prompt
    
    ask(prompt, chat)

    llm_message = answer(
        conv=chat, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt, print_res=print_res
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(llm_message)
    print(f"GT: {data_sample['QA'][0]['a']}")
    return llm_message

def check_answer_egoschema(pred, qid, ans_dict):
    correct = 0
    answer_content = ans_dict[qid]['content'].lower()
    if answer_content[-1] == ".":
        answer_content = answer_content[:-1]
    if ans_dict[qid]['answer'].lower() in pred.lower():
        flag = True
        for kk in ["(A)", "(B)", "(C)", "(D)", "(E)"]:
            if kk != ans_dict[qid]['answer'].lower() and kk in pred.lower():
                flag = ans_dict
                break
        if flag:
            correct += 1
    elif answer_content in pred.lower():
        correct = 1
    elif answer_content.replace("a ", "") in pred.lower():
        correct = 1
    elif answer_content.replace("an ", "") in pred.lower():
        correct = 1
    return correct

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

# %%
from dataset.hd_utils import HD_transform_padding, HD_transform_no_padding

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)

    if padding:
        frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames

# %%
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
    config_file = "configs/config_mistral_hd.json"
    cfg = Config.from_file(config_file)
    cfg.model.vision_encoder.num_frames = 4
    cfg.model.sticky = args.sticky
    cfg.model.alpha = args.alpha
    cfg.model.tau= args.tau
    cfg.model.baseline = args.baseline
    model = VideoChat2_it_hd_mistral(config=cfg.model)

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
    state_dict = torch.load("/mnt/data-poseidon/saul/Ask-Anything/video_chat2/VideoChat2_HD_stage4_Mistral_7B/videochat2_hd_mistral_7b_stage4.pth", "cpu")

    if 'model' in state_dict.keys():
        msg = model.load_state_dict(state_dict['model'], strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    model = model.to(torch.device(cfg.device))
    model = model.eval()

    # You can find the csv files in https://github.com/imagegridworth/IG-VLM/blob/main/data/multiple_choice_qa/EgoSchema.csv
    with open(args.data_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)

        json_data = []
        ans_dict = {}
        
        for idx, msg in enumerate(reader):
            if idx == 0:
                print(msg)
                continue
                
            video = msg[1] + '.mp4'
            input_str = f"Question: {msg[3].capitalize()}\nOptions:\n"
        
            target_index = -1
            for i, candidate in enumerate(msg[5:]):
                option = chr(ord('A') + i)
                input_str += f"({option}) {candidate}\n"
                if candidate == msg[4]:
                    target_index = i
                
            assert target_index != -1
            correct = chr(ord('A') + target_index)
            
            json_data.append({
                'video': video,
                "QA": [{
                    "i": "",
                    "q": input_str.strip(),
                    "a": f"Answer: ({correct}) {msg[4]}",
                }]
            })

            ans_dict[idx - 1] = {
                'video': video,
                'answer': f"({correct})",
                'content': msg[4],
            }

    if args.baseline:
        output_dir = args.output_dir + f"/nframes_{args.max_int}_baseline"  
    else:
        output_dir = args.output_dir + f"/nframes_{args.max_int}_nchunks_{args.num_samples}_tau_{args.tau}_alpha_{args.alpha}" 
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/preds.json"
        #  position embedding
    num_frame = args.max_int
    resolution = 224
    new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
    model.vision_encoder.encoder.pos_embed = new_pos_emb


    total_num = len(json_data)

    output = ""
    results ={}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    for idx, example in enumerate(tqdm(json_data)):
        if example["video"] not in results:
            start = time.time()
            if args.baseline:
                llm_message = infer_egoschema(
                    example, 
                    model,
                    "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n", 
                    question_prompt="\nOnly give the best option.", 
                    answer_prompt="Best option:(",
                    return_prompt='(',
                    system_q=False,
                    print_res=False,
                    system_llm=False,
                    num_segments=args.max_int,
                    video_path=args.video_folder,
                )
            else:
                llm_message = infer_egoschema_inf(
                    example, 
                    model,
                    "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n", 
                    question_prompt="\nOnly give the best option.", 
                    answer_prompt="Best option:(",
                    return_prompt='(',
                    system_q=False,
                    print_res=False,
                    system_llm=False,
                    num_segments=int((args.max_int*args.num_samples)),
                    num_samples=args.num_samples,
                    video_path=args.video_folder,
                )
            
            duration = time.time() - start
            results[example["video"]] = {"answer": ans_dict[idx]["answer"],
                                        "pred": llm_message}
            total =0
            correct = 0
            if os.path.exists(output_file):
                os.remove(output_file)  # Optional: write an empty list to reset the file

            # Append new results
            with open(output_file, 'a') as output_json_file:
                json.dump(results, output_json_file, indent=4)
            
            for idx1, (key, value) in enumerate(results.items()):
                correct += check_answer_egoschema(value["pred"], idx1, ans_dict)
                total += 1
            print(output_dir, flush=True)
            print("Acc:", correct / total, flush=True)
            print('-' * 20, f'{idx+1}/{total_num} done,', f'cost: {duration:.2f}s', '-' * 20, flush=True)
    results["accuracy"] = correct / total
    
    if os.path.exists(output_file):
        os.remove(output_file)  # Optional: write an empty list to reset the file

    # Append new results
    with open(output_file, 'a') as output_json_file:
        json.dump(results, output_json_file, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)