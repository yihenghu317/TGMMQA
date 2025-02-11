import sys

# sys.path.append('../')
import argparse
import os
import csv
from dataloader.multimodalqa.dataloader import MultiModalQATrainDataset
from model.GraphConstruction import KGConstruction
from model.GraphFramework import GraphFramework
from model.GraphAgent import Agent
import torch 
import numpy as np
import time 
np.random.seed(42)
torch.manual_seed(42) 



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--example_file_train",
                        default="./MMQA/dataset/MMQA_train.jsonl",
                        type=str,
                        help="Path to MultimodalQA train set ",
                        )
    parser.add_argument("--few_shot_path", default="./few_shot_examples", type=str)
    parser.add_argument('--dataset_path', default = "/multimodalqa/dataset/")
    parser.add_argument('--image_final_path', default = "/multimodalqa/final_dataset_images")
    parser.add_argument('--temp_data_path', default = "/temp_data")
    parser.add_argument('--half_prec', action='store_false', default=True, )
    parser.add_argument('--data_idx', type=int, default=0)

    args = parser.parse_args()

    data_idx = args.data_idx
    dataset_path = args.dataset_path
    image_final_path = args.image_final_path
    temp_path = args.temp_data_pa
    th
    text_path = os.path.join(dataset_path, "MMQA_texts.jsonl")
    image_path = os.path.join(dataset_path, "MMQA_images.jsonl")
    image_caption = os.path.join(temp_path, 'image_caption.json')
    table_path = os.path.join(dataset_path, "MMQA_tables.jsonl")
    train_path = os.path.join(dataset_path, "MMQA_train.jsonl")
    dev_path = os.path.join(dataset_path, "MMQA_dev.jsonl")
    train_path = dev_path

    dataset = MultiModalQATrainDataset(
        dev_path, text_path,table_path,image_path, image_final_path, 
        image_caption_path = image_caption, 
        temp_path = temp_path
    )

    data = dataset[data_idx]
    model_id = "llama3-70b-instruct"
    framework = GraphFramework(temp_data_path = args.temp_data_path, few_shot_path = args.few_shot_path, half_prec = args.half_prec, model_id = model_id)
    ans = framework(data)
    print("Answer:", ans)

if __name__ == "__main__":
    main()