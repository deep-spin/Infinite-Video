"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import json
import random
import numpy as np
import json
import pandas as pd
import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import GPUtil
import decord
import cv2
import time
from tqdm import tqdm
import subprocess
from moviepy.editor import VideoFileClip
from moviepy.editor import*
from decord import VideoReader
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



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    # Make all arguments required
    parser.add_argument("--task", required=True, help="model.")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--num-beams", type=int, required=True, help="Number of beams.")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature setting.")
    parser.add_argument("--video-folder", required=True, help="Path to video folder.")
    parser.add_argument("--q-folder", required=True, help="Path to question folder.")
    parser.add_argument('--output-dir', required=True, help="Directory to save the model results JSON.")
    
    # New arguments based on constants
    parser.add_argument("--max_int", type=int, required=True, help="Maximum integer value (e.g., 2048).")
    parser.add_argument("--sticky", action="store_true", help="Set sticky flag (True or False).")
    parser.add_argument("--num_basis", type=int, required=True, help="Number of basis (e.g., 64 or 256).")
    parser.add_argument("--event", action="store_true", help="Set event flag (True or False).")
    parser.add_argument("--tau", type=float, required=True, help="Tau value (e.g., 0.75).")
    parser.add_argument("--alpha", type=float, required=True, help="alpha of weighted average between short and long term memory.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecated)."
    )
    
    
    args = parser.parse_args()
    return args

option_str = {"0": "A",
           "1": "B",
           "2": "C",
           "3": "D",
           "4": "E",
}

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

from moviepy.editor import VideoFileClip


class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def get_context_emb(self, input_text, options, msg, img_list):
        
        prompt_1 = "You are able to understand the visual content that the user provides.Follow the instructions carefully and explain your brief answers with no more than 20 words.###Human: <Video><ImageHere></Video>"
    
        
        prompt_2 = input_text
        prompt_3 = "###Assistant:"

        prompt = prompt_1 + " " + prompt_2 + prompt_3

        # Final prompt construction
        prompt = prompt_1 + prompt_2 + prompt_3 
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
    
    def answer(self, img_list, input_text, options, msg, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
            repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        embs = self.get_context_emb(input_text, options, msg, img_list) 

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
    
    def upload_video_without_audio(self, video_list):
        msg = ""
        new_video = True
        video_embs = []
        print(output_dir)
        for i in range(len(video_list)):
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



def initialize_chat():
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model_config.sticky = args.sticky
    model_config.num_basis = args.num_basis
    model_config.sigmas = None
    model_config.tau = args.tau
    model_config.alpha = args.alpha
    model = model_cls.from_config(model_config).to("cuda")
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda')
    print('Initialization Finished')
    return chat, args

def get_video_fragments(video_path, n_frames, task):
    video = torch.load(video_path)
    if task == "video_llama":
        num_frames = video.shape[1]
        # Generate equally spaced indices
        indices = torch.linspace(0, num_frames - 1, steps=n_frames).long()
    
        # Use the indices to sample the frames
        video_list = video[:, indices].unsqueeze(0).to("cuda")
    else:
        video_list = torch.split(video, n_frames, dim=1)
    return video_list

def process_qa(chat, video_list, question, options, num_beams, temperature, task):
    chat.model.long_memory_buffer = []
    chat.model.temp_short_memory = []
    chat.model.short_memory_buffer = []
    if task == "video_llama":
        img_list, _ = chat.model.encode_videoQformer_visual(video_list)
        img_list = [img_list]
        msg = ""
    else:
        msg, img_list = chat.upload_video_without_audio(video_list)
    llm_message = chat.answer(
        img_list=img_list,
        input_text=question,
        options=options,
        msg=msg,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000
    )[0]
    return llm_message

def process_json_file(file_path, chat, args, video_folder, output_file):
    questions = pd.read_csv(file_path)
    num_frames = args.max_int
    result_data = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            result_data = json.load(f)
    for index, qa_key in questions.iterrows():
        if not str(qa_key["video"]) + "_"  +str(qa_key["qid"]) in list(result_data.keys()):
            # Extract video fragments for the current question
            video_list = get_video_fragments(video_folder + '/' + str(qa_key["video"]) + "/"  +str(qa_key["video"]) + ".pt", num_frames, args.task)

            question = qa_key['question']
            options = [qa_key["a0"], qa_key["a1"], qa_key["a2"], qa_key["a3"], qa_key["a4"]]
            llm_message = process_qa(chat, video_list, question, options, args.num_beams, args.temperature, args.task)
            qa_key['pred'] = llm_message
            result_data[str(qa_key["video"]) + "_" + str(qa_key["qid"])] = {"question": question,
                                            "prediction": llm_message,
                                            "answer": option_str[str(qa_key["answer"])],
                                            "options": options                    

            }
            if os.path.exists(output_file):
                os.remove(output_file)  # Optional: write an empty list to reset the file

            # Append new results
            with open(output_file, 'a') as output_json_file:
                json.dump(result_data, output_json_file, indent=4)

def main(chat, args):

    global output_dir
    if args.task == "video_llama":
        output_dir = args.output_dir + f"/nframes_{args.max_int}_video_llama"
    else:
        output_dir = args.output_dir + f"/nframes_{args.max_int}_nbasis_{args.num_basis}_{'sticky' if args.sticky else 'uniform'}" + "_t_"+ str(args.tau).split(".")[1]  + "_gibbs_alpha_" + str(args.alpha)
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/preds.json"
    
    process_json_file(args.q_folder, chat, args, args.video_folder, output_file)

if __name__ == '__main__':

    config_seed = 42
    setup_seeds(config_seed)
    global chat
    chat, args = initialize_chat()
    num_beams = args.num_beams
    temperature = args.temperature
    global output_dir
    output_dir = args.output_dir
    main(chat, args)


