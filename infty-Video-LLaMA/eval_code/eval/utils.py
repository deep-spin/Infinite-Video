from moviepy.editor import*
from decord import VideoReader
import decord
import torch
import random as rnd
import numpy as np

def video_duration(filename):
    with VideoFileClip(filename) as video:
        fps = video.fps  # frames per second
        
        # Calculate the total number of frames
        total_frames = int(video.duration * fps)
        return video.duration, total_frames
 
def capture_video(video_path, fragment_video_path, per_video_length, n_stage):
    start_time = n_stage * per_video_length
    end_time = (n_stage+1) * per_video_length
    video =CompositeVideoClip([VideoFileClip(video_path).subclip(start_time,end_time)])

    video.write_videofile(fragment_video_path)

    
def load_video(video_path, n_frms=16, height=-1, width=-1, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen
    n_frms = min(n_frms, vlen)
    if sampling == "uniform":
        indices = np.linspace(start, end - 1, n_frms).astype(int).tolist()
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


def parse_video_fragment(video_path, video_length, fragment_video_path, n_stage = 0, n_samples = 1):
    decord.bridge.set_bridge("torch")
    per_video_length = video_length / n_samples
    # cut video from per_video_length(n_stage-1, n_stage)
    capture_video(video_path, fragment_video_path, per_video_length, n_stage)
    return fragment_video_path