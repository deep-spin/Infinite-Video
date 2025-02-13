"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import json
import random
import numpy as np
import json
import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import GPUtil
import decord
from decord import VideoReader, cpu
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
from torchvision import transforms


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
    parser.add_argument("--max_int", type=int, required=True, help="Number of samples (e.g., 128).")
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
        
        prompt_1 = "###Human: <Video><ImageHere></Video>\n"

        prompt_2 = (
        "{input_text}\n"
        "Choose the best answer from the options below:\n"
        "{option0}\n"
        "{option1}\n"
        "{option2}\n"
        "{option3}\n"
    ).format(
        input_text=input_text,
        option0=options[0],
        option1=options[1],
        option2=options[2],
        option3=options[3],
    )


        # Combine the prompt parts
        prompt_3 = "###Assistant:"

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
    
    def upload_video_without_audio(self, video_list, num_samples, num_frames=1):
        msg = ""
        new_video = True
        video_embs = 0
        print(output_dir)
        for i in range(num_samples):
            video_fragment = video_list[i]
            if i ==1:
                new_video=False
            video_fragment = self.vis_processor.transform(video_fragment)
            video_fragment = video_fragment.unsqueeze(0).to(self.device)
            self.model.encode_short_memory_frame(video_fragment, num_frames)
            video_emb, _ = self.model.encode_video(new_video=new_video)
            video_embs = i*video_embs/(i+1) + video_emb/(i+1)
        img_list= [video_embs] 
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
def get_index(bound, fps, max_frame,num_segments, first_idx=0):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices
to_tensor = transforms.ToTensor()
# Define the transform pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor()           # Convert to tensor
])

def read_video(video_path, num_segments, bound=None):
    print("\n\n", video_path)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    images_group = list()
    frame_indices = get_index(bound, fps, max_frame, num_segments, first_idx=0) 
    for frame_index in frame_indices:
        # Extract frame, resize it, and convert to tensor
        img = Image.fromarray(vr[frame_index].numpy())  # Use `.asnumpy()` for Decord VideoReader
        img_tensor = transform_pipeline(img)
        images_group.append(img_tensor)
    
    # Stack tensors to create a batch
    torch_imgs = torch.stack(images_group)
    return torch_imgs
def get_video_fragments(video_path, n_frames, n_samples, task):
    if task == "inf_video_llama":
        video = read_video(video_path, int(n_frames*n_samples)).permute(1,0,2,3)
        video_list = torch.chunk(video, n_samples, dim=1)
    else:   
        video_list = read_video(video_path, n_frames).permute(1,0,2,3).unsqueeze(0).to("cuda")  
    return video_list

def process_qa(chat, video_list, question, options, num_beams, temperature, num_frames, num_samples, task):
    chat.model.long_memory_buffer = []
    chat.model.temp_short_memory = []
    chat.model.short_memory_buffer = []
    if task == "video_llama":
        img_list, _ = chat.model.encode_videoQformer_visual(video_list)
        img_list = [img_list]
        msg = ""
    else:
        msg, img_list = chat.upload_video_without_audio(video_list, num_samples=num_samples, num_frames=num_frames)
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

def process_json_file(file_path, args, video_folder):
    output_dir = args.output_dir + f"/nframes_{args.max_int}_nchunks_{args.n_samples}_nbasis_{args.num_basis}_{'sticky' if args.sticky else 'uniform'}" + "_t_"+ str(args.tau).split(".")[1]  + "_gibbs_alpha_" + str(args.alpha) + "_new" 
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/preds.json"
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            try:
                # Parse each line as a JSON object
                json_object = json.loads(line.strip())
                data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {line.strip()} - {e}")
        result_data = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                result_data = json.load(f)
        i=0
        last_video_prefix = None
        for qa_key in tqdm(data):
            if qa_key["question_id"] not in result_data:
                current_video_prefix = qa_key["video_id"]  # Extract the first three letters of the current videoID

                # Check if the current prefix matches the last processed one
                if current_video_prefix != last_video_prefix:
                    i += 1
                    print(output_file)

                    # Extract video fragments for the current question
                    video_list = get_video_fragments(
                        video_folder + '/' + qa_key["videoID"] + ".mp4",
                        args.max_int,
                        args.n_samples,
                        args.task
                    )
                    # Update the last processed prefix
                    last_video_prefix = current_video_prefix
                if i==1:
                    global chat
                    chat, args = initialize_chat(args)
                question = qa_key['question']
                options = qa_key["options"]
                llm_message = process_qa(chat, video_list, question, options, args.num_beams, args.temperature, args.max_int, args.n_samples, args.task)
                qa_key['pred'] = llm_message
                result_data[qa_key["question_id"]] = {"question": question,
                                                "prediction": llm_message,
                                                "answer": qa_key["answer"],
                                                "options": options,
                                                "duration": qa_key["duration"]

                }

                if os.path.exists(output_file):
                    os.remove(output_file)  # Optional: write an empty list to reset the file

                # Append new results
                with open(output_file, 'a') as output_json_file:
                    json.dump(result_data, output_json_file, indent=4)

def main():
    global chat
    args = parse_args()
    num_beams = args.num_beams
    temperature = args.temperature
    global output_dir
    output_dir = args.output_dir
    
    process_json_file(args.q_folder, args, args.video_folder)

if __name__ == '__main__':

    config_seed = 42
    setup_seeds(config_seed)
    main()


