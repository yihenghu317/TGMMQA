
import sys
sys.path.append('../')
sys.path.append('/home/hu052/data_discovery/MMQA')
from evaluate.eval_function import evaluate_predictions
from evaluate.common_utils import MULTI_HOP_QUESTION_TYPES
from collections import Counter


import argparse
import os
import csv
from dataloader.multimodalqa.dataloader import MultiModalQATrainDataset
# from model.preGraphConstruction import KGConstruction
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
    parser.add_argument('--dataset_path', default = "dataset/")
    parser.add_argument('--image_final_path', default = "final_dataset_images")
    parser.add_argument('--temp_data_path', default = "temp_data")
    parser.add_argument('--half_prec', action='store_false', default=True, )
    parser.add_argument('--data_idx', type=int, default=0)

    args = parser.parse_args()

    data_idx = args.data_idx
    dataset_path = args.dataset_path
    image_final_path = args.image_final_path
    temp_path = args.temp_data_path
    text_path = os.path.join(dataset_path, "MMQA_texts.jsonl")
    image_path = os.path.join(dataset_path, "MMQA_images.jsonl")
    image_caption = os.path.join(temp_path, 'image_caption.json')
    table_path = os.path.join(dataset_path, "MMQA_tables.jsonl")
    train_path = os.path.join(dataset_path, "MMQA_train.jsonl")
    dev_path = os.path.join(dataset_path, "MMQA_dev.jsonl")

    dataset = MultiModalQATrainDataset(
        dev_path, text_path,table_path,image_path, image_final_path, 
        image_caption_path = image_caption
    )
    
    id_answer_dict = {}

    with open('answer.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            id_answer_dict[row['ID']] = row['Answer']

    predicted_answers = id_answer_dict


    gold_answers, answer_modalities, hop_types, question_types = {}, {}, {}, {}
    for data in dataset:
        qid = data['meta']['qid']
        example = data['meta']

        gold_answer = [str(item["answer"]) for item in example["answers"]]
        gold_answers[qid] = [gold_answer]
        answer_modality = set([item["modality"] for item in example["answers"]])
        assert len(answer_modality) == 1
        answer_modalities[qid] = answer_modality.pop()
        question_types[qid] = example["metadata"]["type"]
        hop_types[qid] = "Multi-hop" if example["metadata"]["type"] in MULTI_HOP_QUESTION_TYPES else "Single-hop"
    
    eval_scores, instance_eval_results = evaluate_predictions(predicted_answers, gold_answers)
    print("\n\nOverall result with different metrics: ")
    for metric, value in eval_scores.items():
        print(f"{metric}: {value}")

    modality_counts = Counter(answer_modalities.values())
    _, _, eval_scores_by_modalities = \
        evaluate_predictions(predicted_answers, gold_answers, answer_modalities)
    print("\n\nEval results for different modalities:")
    for answer_modality in sorted(eval_scores_by_modalities.keys()):
        result = eval_scores_by_modalities[answer_modality]
        print(f"{answer_modality}")
        print(f"# of examples: {modality_counts[answer_modality]}")
        for metric, value in result.items():
            print(f"{metric}: {value}")

    hop_type_counts = Counter(hop_types.values())
    _, _, eval_scores_by_hop_types = evaluate_predictions(predicted_answers, gold_answers, hop_types)
    print("\n\nType\tCount\tEM\tF1")
    for hop_type in sorted(eval_scores_by_hop_types.keys()):
        result = eval_scores_by_hop_types[hop_type]
        print(f"{hop_type}\t{hop_type_counts[hop_type]}\t{result['list_em']}\t{result['list_f1']}")

    question_type_counts = Counter(question_types.values())
    _, _, eval_scores_by_qtypes = evaluate_predictions(predicted_answers, gold_answers, question_types)
    print("\n\nType\tCount\tEM\tF1")
    for question_type in sorted(eval_scores_by_qtypes.keys()):
        result = eval_scores_by_qtypes[question_type]
        print(f"{question_type}\t{question_type_counts[question_type]}\t{result['list_em']}\t{result['list_f1']}")


 


if __name__ == "__main__":
    main()