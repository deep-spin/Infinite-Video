import os
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--input_path", required=True, help="videos path")
    parser.add_argument("--output_path", required=True, type=str, help="path to save features.")
    parser.add_argument("--num_frames", required=True, type=int, help="number of frames to sample from each video.")
    
    args = parser.parse_args()
    return args

def extract_and_save_features(input_path, output_path, num_frames):
    input_base_path = Path(input_path)
    output_base_path = Path(output_path)

    # Placeholder for answers if needed later
    answers = {}

    pbar = tqdm(total=len(list(input_base_path.iterdir())))
    for video_fp in list(input_base_path.iterdir()):
        if video_fp.stem not in [p.stem for p in output_base_path.iterdir()]:
            output_path = output_base_path / video_fp.stem
            output_path.mkdir(parents=True, exist_ok=True)

            # Use OpenCV to read video frames efficiently
            video_capture = cv2.VideoCapture(str(str(video_fp)))
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # Uniformly sample frame indices
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

            video_frames = []
            for frame_idx in frame_indices:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = video_capture.read()
                if success:
                    frame = cv2.resize(frame, (224, 224))
                    video_frames.append(frame)
            video_capture.release()

            # Save features as images
            save_image_frames(video_frames, video_fp.stem, output_path)
        pbar.update(1)

    pbar.close()

def save_image_frames(video_frames, name_ids, save_folder):
    """
    Save video frames as image files in a specified folder.

    Args:
    - video_frames (list): List containing video frames
    - name_ids (str): Identifier to include in the filename
    - save_folder (str): Path to the folder where the images should be saved

    Returns:
    - None
    """
    for idx, frame in enumerate(video_frames):
        filename = f"{name_ids}_frame_{idx:04d}.jpg"  # Construct filename with frame index
        filepath = os.path.join(save_folder, filename)
        cv2.imwrite(filepath, frame)  # Save frame as image

if __name__ == '__main__':
    args = parse_args()
    extract_and_save_features(args.input_path, args.output_path, args.num_frames)