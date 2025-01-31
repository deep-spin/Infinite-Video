"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from InfVideoLLaMA.common.config import Config
from InfVideoLLaMA.common.dist_utils import get_rank
from InfVideoLLaMA.common.registry import registry
from InfVideoLLaMA.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle
import decord
import cv2
import time
import subprocess
from moviepy.editor import VideoFileClip
from decord import VideoReader
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from InfVideoLLaMA.datasets.builders import *
from InfVideoLLaMA.models import *
from InfVideoLLaMA.processors import *
from InfVideoLLaMA.runners import *
from InfVideoLLaMA.tasks import *
from moviepy.editor import*

import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import GPUtil
import gradio as gr

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu", type=str, default="cuda", help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--text-query", required=True, help="question the video")
    parser.add_argument("--video-path", required=True, help="path to video file.")
    parser.add_argument("--fragment-video-path", required=True, help="path to video fragment file.")
    
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
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config_seed):
    seed = config_seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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


def video_duration(filename):
    with VideoFileClip(filename) as video:
        fps = video.fps  # frames per second
        
        # Calculate the total number of frames
        total_frames = int(video.duration * fps)
        return video.duration
 
def capture_video(video_path, fragment_video_path, per_video_length, n_stage):
    start_time = n_stage * per_video_length
    end_time = (n_stage+1) * per_video_length
    video =CompositeVideoClip([VideoFileClip(video_path).subclip(start_time,end_time)])
    video.write_videofile(fragment_video_path)

    
def load_video(video_path, n_frms=256, height=-1, width=-1, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width, num_threads=1)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)
    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices)
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


def parse_video_fragment(video_path, video_length, n_stage = 0, n_samples = 8):
    decord.bridge.set_bridge("torch")
    per_video_length = video_length / n_samples
    # cut video from per_video_length(n_stage-1, n_stage)
    capture_video(video_path, fragment_video_path, per_video_length, n_stage)
    return fragment_video_path

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.output_text = " "
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def get_context_emb(self, input_text, msg, img_list):
        
        prompt_1 = "###Human: <Video><ImageHere></Video>"
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
    
    def gradio_answer(self,chatbot, chat_state):
        import pdb;pdb.set_trace()
        return gr.update(value=self.output_text, interactive=False),None

    def answer(self, img_list, input_text, msg, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.90,
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

    
    def parse_video(self, video_path, video_length, n_stage=0, n_samples= 8):
        video_fragment = parse_video_fragment(video_path=video_path, video_length=video_length, n_stage=n_stage, n_samples= n_samples)
        video_fragment, msg = load_video(
            video_path=fragment_video_path,
            n_frms=args.max_int, 
            height=224,
            width=224,
            sampling ="uniform", return_msg = True
        ) 
        video_fragment = self.vis_processor.transform(video_fragment)
        video_fragment = video_fragment.unsqueeze(0).to(self.device)
        return video_fragment
    
    def upload_video_without_audio(self, args, video_list, video_path, num_frames):
        msg = ""
        new_video = True
        if isinstance(video_path, str):  # is a video path
            print(video_path)
            video_emb = 0
            video_list = torch.chunk(video_list, num_frames, dim=1)
            for i in range(num_frames): # 28
                video_fragment = video_list[i]
                if i ==1:
                    new_video=False
                video_fragment = self.vis_processor.transform(video_fragment)
                video_fragment = video_fragment.unsqueeze(0).to(self.device)
                self.model.encode_short_memory_frame(video_fragment, args.max_int)
                video_embs, _ = self.model.encode_video(new_video=new_video)
                video_emb = i*video_emb/(i+1) + video_embs/(i+1)   # Shape: [1, 32, 4096]
            
            img_list= [video_emb]
            return msg, img_list 
                
        return msg, img_list  

def initialize_chat(args):
    print('Initializing Chat')
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

def save_image_features(img_feats, name_ids, save_folder):
    """
    Save image features to a .pt file in a specified folder.

    Args:
    - img_feats (torch.Tensor): Tensor containing image features
    - name_ids (str): Identifier to include in the filename
    - save_folder (str): Path to the folder where the file should be saved

    Returns:
    - None
    """
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = os.path.join(save_folder, filename)
    torch.save(img_feats, filepath)

if __name__ =='__main__':
    config_seed = 42
    args = parse_args()
    setup_seeds(config_seed)
    video_path = args.video_path
    fragment_video_path = args.fragment_video_path
    video_length = video_duration(video_path) 
    video_fragment = parse_video_fragment(video_path=video_path, video_length=video_length, n_stage=0, n_samples= 1)
    video_list, msg = load_video(
        video_path=video_fragment,
        n_frms=int(args.max_int*args.n_samples), 
        height=224,
        width=224,
        sampling ="uniform", return_msg = True
        )
    save_image_features(video_list,video_path.replace(".mp4", "").split("/")[-1] + "3_chunks", os.path.dirname(video_path))
    chat, args = initialize_chat(args)

    msg, img_list = chat.upload_video_without_audio(args, video_list=video_list,
        video_path=video_path, 
        num_frames=args.n_samples
        )
    
    text_input = args.text_query

    num_beams = args.num_beams
    temperature = args.temperature
    llm_message = chat.answer(img_list=img_list,
                              input_text=text_input,
                              msg = msg,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]

    print(llm_message)
    # File path
    file_path = args.text_query + "bohemian_sticky_"+".txt"

    # Save the content to a text file
    with open(file_path, "w") as file:
        file.write(llm_message)
