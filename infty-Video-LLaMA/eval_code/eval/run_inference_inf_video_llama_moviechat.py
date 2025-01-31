"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import json
import random
import numpy as np
import json
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import GPUtil
import decord
import cv2
import time
from tqdm import tqdm
import subprocess
decord.bridge.set_bridge('torch')

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os

current_dir = os.getcwd()
sys.path.append(current_dir)

from InfVideoLLaMA.datasets.builders import *
from InfVideoLLaMA.models import *
from InfVideoLLaMA.processors import *
from InfVideoLLaMA.runners import *
from InfVideoLLaMA.tasks import *
from InfVideoLLaMA.common.config import Config
from InfVideoLLaMA.common.dist_utils import get_rank
from InfVideoLLaMA.common.registry import registry
from InfVideoLLaMA.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle

from utils import video_duration, parse_video_fragment, load_video
import warnings

# Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    # Make all arguments required
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--num-beams", type=int, required=True, help="Number of beams.")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature setting.")
    parser.add_argument("--video-folder", required=True, help="Path to video folder.")
    parser.add_argument("--qa-folder", required=True, help="Path to GT (ground truth) folder.")
    parser.add_argument('--output-dir', required=True, help="Directory to save the model results JSON.")
    # New arguments based on constants
    parser.add_argument("--max_int", type=int, required=True, help="Maximum integer value (e.g., 2048).")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples (e.g., 128).")
    parser.add_argument("--sticky", action="store_true", help="Set sticky flag (True or False).")
    parser.add_argument("--num_basis", type=int, required=True, help="Number of basis (e.g., 64 or 256).")
    parser.add_argument("--tau", type=float, required=True, help="Tau value (e.g., 0.75).")
    parser.add_argument("--alpha", type=float, required=True, help="alpha of weighted average between short and long term memory.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecated)."
    )
    
    
    args = parser.parse_args()
    return args


def setup_seeds(config_seed):
    seed = config_seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def get_context_emb(self, input_text, msg, img_list):
        
        # prompt_1 = "You are able to understand the visual content that the user provides.Follow the instructions carefully and explain your brief answers with no more than 20 words.###Human: <Video><ImageHere></Video>"
        prompt_1 = "You are able to understand the visual content that the user provides.Follow the instructions carefully and explain your answers.###Human: <Video><ImageHere></Video>"
        
        prompt_2 = input_text
        prompt_3 = "###Assistant:"

        prompt = prompt_1 + " " + prompt_2 + prompt_3

        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    
    def answer(self, img_list, input_text, msg, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
            repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        embs = self.get_context_emb(input_text, msg, img_list) 

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]
        
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
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
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        return output_text, output_token.cpu().numpy()
    
    def upload_video_without_audio(self, video_list, num_frames=1):
        msg = ""
        new_video = True
        video_embs = []
        print(output_dir)
        for i in range(num_frames):
            video_fragment = video_list[i]
            if i ==1:
                new_video=False
            video_fragment = self.vis_processor.transform(video_fragment)
            video_fragment = video_fragment.unsqueeze(0).to(self.device)
            self.model.encode_short_memory_frame(video_fragment, args.max_int)
            video_emb, _ = self.model.encode_video(new_video=new_video)
            video_embs.append(video_emb)
            
        video_emb = torch.mean(torch.stack(video_embs), dim=0, keepdim=True).squeeze(0)  # Shape: [1, 32, 4096]
        img_list= [video_emb] 
        return msg, img_list   



def initialize_chat(args):
    print('Initializing Chat')
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model_config.sticky = args.sticky
    model_config.num_basis = args.num_basis
    model_config.tau = args.tau
    model_config.alpha = args.alpha
    model = model_cls.from_config(model_config).to("cuda")
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda')
    print('Initialization Finished')
    return chat, args

def get_video_fragments(video_path, n_frames):
    video = torch.load(video_path)
    video_list = torch.split(video, n_frames, dim=1)
    return video_list

def process_qa(chat, video_list, question, num_beams, temperature, num_frames):
    chat.model.long_memory_buffer = []
    chat.model.temp_short_memory = []
    chat.model.short_memory_buffer = []
    msg, img_list = chat.upload_video_without_audio(video_list, num_frames=num_frames)
    llm_message = chat.answer(
        img_list=img_list,
        input_text=question,
        msg=msg,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000
    )[0]
    return llm_message

def process_json_file(file_path, chat, args, video_folder, output_file):
    with open(file_path, 'r') as json_file:
        movie_data = json.load(json_file)
        global_key = movie_data["info"]["video_path"]
        result_data = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                result_data = json.load(f)
        if movie_data["info"]["video_path"] not in list(result_data.keys()):
            print(output_file)
            video_path = os.path.join(video_folder, movie_data["info"]["video_path"])
            video_list = get_video_fragments(video_path.replace(".mp4", "") + "/" + movie_data["info"]["video_path"].replace(".mp4", "") + ".pt" , args.max_int)
            
            global_value = []
            qa_data = movie_data["global"]
            
            for qa_key in qa_data:
                question = qa_key['question']
                llm_message = process_qa(chat, video_list, question, args.num_beams, args.temperature, args.n_samples)
                qa_key['pred'] = llm_message
                global_value.append(qa_key)
            
            result_data[global_key]= global_value

            if os.path.exists(output_file):
                os.remove(output_file)  # Optional: write an empty list to reset the file

            with open(output_file, 'a') as output_json_file:
                json.dump(result_data, output_json_file, indent=4)

def main(chat, args):

    global output_dir
    output_dir = args.output_dir + f"/nframes_{args.max_int}_nchunks_{args.n_samples}_nbasis_{args.num_basis}_{'sticky' if args.sticky else 'uniform'}" + "_t_"+ str(args.tau).split(".")[1]  + "_gibbs_alpha_" + str(args.alpha)
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/preds.json"
    json_files = [f for f in os.listdir(args.qa_folder) if f.endswith('.json')]
    for file in json_files:
        file_path = os.path.join(args.qa_folder, file)
        process_json_file(file_path, chat, args, args.video_folder, output_file)
if __name__ == '__main__':

    config_seed = 42
    setup_seeds(config_seed)
    global chat
    args = parse_args()
    chat, args = initialize_chat(args)
    num_beams = args.num_beams
    temperature = args.temperature
    video_folder = args.video_folder
    qa_folder = args.qa_folder
    global output_dir
    output_dir = args.output_dir

    main(chat, args)


