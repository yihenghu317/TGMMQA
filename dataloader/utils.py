import torch
from torch.utils.data import Dataset
import jsonlines
import pickle
import numpy as np
import os 


def read_jsonlines(file_name):
    lines = []
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

def construct_table_str(data):
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

def extract_full_supporting_context(data):
    image = {doc["id"]: doc for doc in data['image_content']}
    text = {doc["id"]: doc for doc in data['text_content']}
    table = {doc["id"]: doc for doc in data['table_content']}
    full_info = {}
    for context in data['meta']['supporting_context']: 
        try:
            if context['doc_part'] == 'image':
                full_info[context['doc_id']] = image[context['doc_id']]
                full_info[context['doc_id']]['modality'] = 'image'
            
            if context['doc_part'] == 'table':
                full_info[context['doc_id']] = table[context['doc_id']]
                full_info[context['doc_id']]['modality'] = 'table'
            
            if context['doc_part'] == 'text':
                full_info[context['doc_id']] = text[context['doc_id']]
                full_info[context['doc_id']]['modality'] = 'text'

        except:
            print("An exception occurred!!!!: "  + data['meta']['qid'])
        
    return full_info

def edit_supporting_context(data):
    contexts = extract_full_supporting_context(data)
    context_str = "Context:\n"
    for key, context in contexts.items():
        # process by modality 
        if context['modality'] == 'table':
            context_str += construct_table_str(context) + "Table End.\n"
            continue

        if context['modality'] == 'text':
            context_str += context['text'] + "\n"
            continue 

        if context['modality'] =='image':
            context_str += "The title of image is " + context['title'] + '.\n'
            continue
    

    return context_str

# 'supporting_context': [{'doc_id': 'f219f551340e5fc9d7b83ae9d5e6bef6', 'doc_part': 'table'}, {'doc_id': 'f721969ee3c7638835a273bf7cd257dc', 'doc_part': 'image'}, {'doc_id': '593524a191f539f2df570f818e991489', 'doc_part': 'image'}]}

# ['172af91df9873976899be2fc8a4e3d82', '05fa6a133c6f088f7defcb5bfa639f75', '89a09361092cc721019083e57cbfac6d', 'f99ab2bb590a363e4b946f2acf1f8a1b', '9ea03eeba5ad182ec3ee3a72559213af', 'ee855d382b1c84031f7554b4154aaf29', '1af90f364f1cea69efbbbd95d72c8095'],


def construct_full_table_str(data):
    col_name = []
    for col in data['table']['header']:
        col_name.append(col['column_name'])

    table_header = f"In table of {data['title']}, "
    table_str = f"In table of {data['title']}:\n"
    table_str = ""
    for i,row in enumerate(data['table']['table_rows']):
        # row_str = f"In row{i} : "
        row_str = f"In row{i} of table {data['title']}: "
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