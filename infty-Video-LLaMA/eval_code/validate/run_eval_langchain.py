import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
import sys
import os
from openai import OpenAI
import shutil
import random
from utils import *
from multiprocessing import Manager
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
embedding = OpenAIEmbeddings(openai_api_key="")


manager = Manager()
results = manager.dict()

option_str = {"0": "A",
           "1": "B",
           "2": "C",
           "3": "D",
           "4": "E",
}

current_dir = os.getcwd()
sys.path.append(current_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args

def annotate(prediction_set, keys_chunk):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    for i, key in enumerate(keys_chunk):
        id = key
        key = prediction_set[key]
        # Define the five prediction options
        predictions = [
            {"prediction": key["options"][0]},
            {"prediction": key["options"][1]},
            {"prediction": key["options"][2]},
            {"prediction": key["options"][3]},
            #{"prediction": key["options"][4]},
        ]

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=predictions,
            embeddings=embedding,  # Embedding model for similarity
            vectorstore_cls=Chroma,  # VectorStore for indexing and searching
            k=1,  # Return the single most similar prediction,
            collection_name = "collection" + str(random.randint(1, 1000000000))
        )

                # Store the ground truth separately
        ground_truth = {"prediction": key["prediction"]} 

        # Select the most similar prediction using the ground truth
        pred = example_selector.select_examples(ground_truth) 
        index = key["options"].index(pred[0]["prediction"])
        answer = option_str[str(index)]
        if "duration" in key:
            results[id] = {"prediction": answer,
                       "answer": key["answer"],
                        "duration": key["duration"]}
        else:
            results[id] = {"prediction": answer,
                       "answer": key["answer"]}

def main(args):
    """
    Main function to control the flow of the program.
    """

    prediction_set = load_json(args.pred_path)
    num_tasks = args.num_tasks
    # import pdb; pdb.set_trace()
    ids = list(prediction_set.keys())

    # Split tasks into parts.
    part_len = len(ids) // num_tasks
    all_parts = [ids[i:i + part_len] for i in range(0, len(ids), part_len)]
    task_args = [(prediction_set, part) for part in all_parts]

    # Use a pool of workers to process the files in parallel.
    with Pool() as pool:
        pool.starmap(annotate, task_args)

if __name__ == "__main__":
    args = parse_args()
    # Get the directory name
    directory = os.path.dirname(args.pred_path)

    # Create the new directory path with 'results_' prefix
    output_dir = os.path.join(os.path.dirname(directory), "results_" + os.path.basename(directory))
    print(output_dir)
    args.output_dir= output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        # Create the directory again (empty)
        os.makedirs(args.output_dir)
    
    main(args)
    save_json(dict(results), output_dir + "/preds.json")
