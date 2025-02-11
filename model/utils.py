import torch
from torch.utils.data import Dataset
import jsonlines
import pickle
import numpy as np
import os 
import ast

import re

import base64
from PIL import Image
from PIL import ImageFile
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True
def image_reader(fp, line):
    fp.seek(line)
    imgid, img_base64 = fp.readline().strip().split('\t')
    img_feat = Image.open(BytesIO(base64.b64decode(img_base64)))
    return img_feat


def read_jsonlines(file_name):
    lines = []
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def bag_of_words(string, splitter = '_'):
    bow = list(string.lower().split(splitter))
    results = set() 
    for word in bow:
        pre = re.sub(r'[^a-zA-Z0-9_]', '', word)
        results.add(pre)

    return results

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))

def process_bag_of_words(string, splitter='_'):
    # wordsList = string.lower().replace(splitter, ' ')
    wordsList = string.lower().replace(splitter, ' ')
    wordsList = nltk.word_tokenize(wordsList)
    wordsList = [w for w in wordsList if not w in stop_words]
    tagged = nltk.pos_tag(wordsList)
    results = set()
    for tag, pos in tagged:
        # if "NN" in pos:
        results.add(tag)
    return results


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    return input

def extract_question_modality(text):
    # Define the regex patterns to extract the Question and Modality
    question_pattern = r"(?i)question:\s*(.*?)(?=\s*modality:|$)"
    modality_pattern = r"(?i)modality:\s*(.*)"

    # Search for matches in the input text
    question_match = re.search(question_pattern, text)
    modality_match = re.search(modality_pattern, text)

    # Extract the groups if matches are found
    question = question_match.group(1).strip() if question_match else None
    modality = modality_match.group(1).strip() if modality_match else None

    return question, modality

def construct_full_table_str(data):
    col_name = []
    for col in data['table']['header']:
        col_name.append(col['column_name'])

    table_header = f"In table of {data['title']}, "
    table_str = f"In table of {data['title']}:\n"
    # table_str = ""
    for i,row in enumerate(data['table']['table_rows']):
        row_str = f"In row{i} : "
        # row_str = table_header
        for j, col in enumerate(row): 
            if len(col['text']) == 0: 
                continue
            row_str += f"{col_name[j]} is {col['text']}"
            if j < len(row) - 1: 
                row_str += ", "
            else:
                row_str += "."
        row_str += "\n"

        table_str +=  row_str
        
    return table_str


def string_to_list(input_string):
    pattern = r"\[.*\]"
    match = re.search(pattern, input_string)

    if match:
        # Extract the matched string
        list_str = match.group(0)
        
        # Convert the string representation of the list into an actual Python list
        extracted_list = ast.literal_eval(list_str)
        
        # print(extracted_list)
        return extracted_list
    else:
        print("No list found in the string.")
        return input_string
