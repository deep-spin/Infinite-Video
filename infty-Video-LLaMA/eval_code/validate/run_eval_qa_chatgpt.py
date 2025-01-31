"""
Adapted from: https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/quantitative_evaluation/evaluate_activitynet_qa.py
"""
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys
import os
import torch
from openai import OpenAI
from utils import promp_selector
import csv
import shutil

current_dir = os.getcwd()
sys.path.append(current_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args


def annotate(metric, prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    for file in caption_files:
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        try:
            # Compute the correctness score
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages= promp_selector(metric, question, answer, pred)
            )
            # Convert response to a Python dictionary.
            response_message = completion.choices[0].message.content
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]
            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")

def save_results_to_csv(output_dir, results):
    # CSV file path
    csv_file = os.path.join(output_dir, "results.csv")
    
    # Write the results to the CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing the header row
        writer.writerow(metrics)  # Write the metric names as the header (6 columns)

        # Writing accuracy row
        accuracy_row = [results[metric][0] if results[metric][0] is not None else "None" for metric in metrics]
        writer.writerow(accuracy_row)

        # Writing score row
        score_row = [results[metric][1] for metric in metrics]
        writer.writerow(score_row)

def main(args):
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    pred_contents = []
    with open(args.pred_path, 'r') as file:
        try:
            # Load the entire JSON file as a dictionary
            json_object = json.load(file)
            
            # Iterate through the dictionary and aggregate the lists
            for key, value in json_object.items():
                if key not in pred_contents:
                    pred_contents.append({key: value})
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON - {e}")
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []
    if len(pred_contents)==1:
        for sample in pred_contents[0]:
            # Create a new sample with the modified key
            new_sample = sample
            new_sample['id'] = sample["id"]
            new_pred_contents.append(new_sample)
    else:
        # Iterate through each sample in pred_contents
        for clip in pred_contents:
            video_id = list(clip.keys())[0]
            for sample in clip[video_id]:
                if video_id in video_id_counts:
                    video_id_counts[video_id] += 1
                else:
                    video_id_counts[video_id] = 0

                # Create a new sample with the modified key
                new_sample = sample
                new_sample['id'] = f"{video_id}_{video_id_counts[video_id]}"
                new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files

    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]
    # import pdb; pdb.set_trace()
    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        # import pdb; pdb.set_trace()
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set
    global client
    # Set the OpenAI API key.
    client = OpenAI(api_key="your api")
    # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url="https://api.aiproxy.io/v1")'
    # openai.api_base = "https://api.aiproxy.io/v1"
    num_tasks = args.num_tasks
    # import pdb; pdb.set_trace()

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(args.metric, prediction_set, part, args.output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:

                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_dir + "/acc.json"  

    # Iterate through json files
    for file_name in os.listdir(args.output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(args.output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in combined_contents.items():
        # Computing score
        count += 1
        score_match = result[0]['score']
        score = int(score_match)
        score_sum += score
        
        if args.metric == "GEN":
            # Computing accuracy
            pred = result[0]['pred']
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1

    if args.metric == "GEN":
        accuracy = yes_count / (yes_count + no_count)
        print("Yes count:", yes_count)
        print("No count:", no_count)
        print("Accuracy:", accuracy)
    average_score = score_sum / count    
    print("Average score:", average_score)
    output_file_path = os.path.join(args.output_dir, "results.txt")

    # Writing the results to a file in the output directory
    with open(output_file_path, 'w') as f:
        f.write(f"Average Score: {average_score}\n")
        if args.metric == "GEN":
            f.write(f"Accuracy: {accuracy}\n")
    if args.metric != "GEN":
        return None, average_score
    else:
        return accuracy, average_score
if __name__ == "__main__":
    args = parse_args()
    metrics = ["GEN", "CI", "DO", "CU"]
    # Get the directory name
    directory = os.path.dirname(args.pred_path)

    # Create the new directory path with 'results_' prefix
    output_dir = os.path.join(os.path.dirname(directory), "results_" + os.path.basename(directory))
    print(output_dir)
    args.output_dir= output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        # Create the directory again (empty)
        os.makedirs(args.output_dir)
    results = {}
    dir = args.output_dir
    # Loop over the metrics and calculate accuracy and score
    for metric in metrics:
        # Set the current metric in args
        args.metric = metric
        
        # Update the output directory for the specific metric
        args.output_dir = os.path.join(dir, metric)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Calculate acc and score for the current metric
        acc, score = main(args)
        
        # Store the results in the dictionary
        results[metric] = (acc, score)
    
    # Save results to a CSV file in the main output directory
    save_results_to_csv(dir, results)
    