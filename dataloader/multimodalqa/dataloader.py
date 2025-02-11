import torch
from torch.utils.data import Dataset
import jsonlines
import pickle
import numpy as np
import os 
import json
from dataloader.utils import edit_supporting_context, construct_full_table_str

# different modalities 
def read_jsonlines(file_name):
    lines = []
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


class MultiModalQATrainDataset(Dataset): 
    def __init__(self, 
        train_path, 
        text_path, 
        table_path,   
        image_meta_path, 
        image_path, 
        image_caption_path = '/home/hu052/data_discovery/temp_data/image_caption.json', 
        temp_path = './',
    ):
        self.train_info = read_jsonlines(train_path)
        image_info = read_jsonlines(image_meta_path)
        text_data = read_jsonlines(text_path)
        table_data = read_jsonlines(table_path)

        self.texts_dict = {doc["id"]: doc for doc in text_data}
        self.image_dict = {doc["id"]: doc for doc in image_info}
        self.table_dict = {doc["id"]: doc for doc in table_data}
    
        self.image_path = image_path

        f = open(image_caption_path)
        image_cap_info = json.load(f)
        self.image_caption_dict = {doc['id']:doc for doc in image_cap_info}

        self.temp_path = temp_path


    def __getitem__(self, index):
        """
        
        metadata: dict_keys(['type', 'modalities', 'wiki_entities_in_question', 'wiki_entities_in_answers', 'pseudo_language_question', 'rephrasing_meta', 'image_doc_ids', 'text_doc_ids', 'intermediate_answers', 'table_id'])
        Return: question, answer, entity, modality        
        
        
        """
        
        example = self.train_info[index]
        qid = example['qid']
        # ['qid', 'question', 'answers', 'metadata', 'supporting_context']
        # print(example.keys())
        # print(example['metadata'])
        # print(example['supporting_context'])
        # metadata: ['wiki_entities_in_question', 'wiki_entities_in_answers', 'type', 'modalities', 'pseudo_language_question', 'rephrasing_meta', 'image_doc_ids', 'text_doc_ids', 'intermediate_answers', 'table_id']
        # metadata example: {'wiki_entities_in_question': [], 'wiki_entities_in_answers': [{'text': '2018', 'wiki_title': 'United States House of Representatives elections, 2018', 'url': 'https://en.wikipedia.org/wiki/United_States_House_of_Representatives_elections,_2018', 'table_cells': [[10, 0]]}], 'type': 'TableQ', 'modalities': ['table'], 'pseudo_language_question': 'In [Electoral history] of [Dutch Ruppersberger], what was the [Year](s) when the [Opponent] was [Elizabeth Matory]', 'rephrasing_meta': {'accuracy': 0.8724749560911659, 'edit_distance': 0.8977272727272727, 'confidence': 0.9981946061810767}, 'image_doc_ids': ['66b4d9213088ab9ce0d7a55655ee7c98', '82fc3d787bf62019be2f1f10d69bc170', 'd9596e306ec297784b87c0321bb9bedf', '6f6767c28cf2de8a701cb866e3a3e190', '35b31d9b4f723f806fd32662ef29edf7'], 'text_doc_ids': ['64393fd65df3168ebe0c26f694563e00', 'df12c2fccc233690ec7a1d2d57ef26c5', '324bdefbdd04ae7c1d4b6643b6e3c626', '5c0d2f87c3c54e7c058099c534ddfeec', '8639dea34996ce2e64a03dcc16c8a81d', '0c6b8541fd06d5d565582b2d2bad1670', '974e568542dc23174fdc2dd7f3685b03', 'b5c07b129592064b8294d57291a2fe2d', '4ee001b77da93f92826bf99f1119c73f', '345d8679d02b36973337ca28dd9c1681'], 'intermediate_answers': [], 'table_id': 'dcd7cb8f23737c6f38519c3770a6606f'}
        
        text_list = []
        for docid in example['metadata']['text_doc_ids']:
            text_info = self.texts_dict[docid]
            text_info['modality'] = 'text'
            text_list.append(text_info)
        
        image_list = []
        image_caption_list = []
        for imgid in example['metadata']['image_doc_ids']: 
            img_info = self.image_dict[imgid] 
            img_info['path'] = os.path.join(self.image_path,img_info['path'])
            img_info['modality'] = 'image'
            image_list.append(img_info)

            if imgid in self.image_caption_dict.keys():
                img_cap_info = self.image_caption_dict[imgid]
                image_caption_list.append(img_cap_info)
            
        image_embed = []
        if len(image_list) > 0: 
            image_embed_path = os.path.join(self.temp_path, f'multimodal_image_embed/{qid}.npy')
            if os.path.exists(image_embed_path):
                try:
                    image_embed = np.load(image_embed_path)
                    image_embed = image_embed
                except:
                    print(f"Something wrong with image embed: {qid}")    
        
        triple_embed = []
        triple_embed_path = os.path.join(self.temp_path, f'multimodal_triple_embed/{qid}.npy')
        if os.path.exists(triple_embed_path):
            try:
                triple_embed = np.load(triple_embed_path, allow_pickle=True).item()
            except:
                print(f"Something wrong with triple embed: {qid} | {triple_embed_path}")


        table_list = []
        if 'table_id' in example['metadata'].keys():
            table = self.table_dict[example['metadata']['table_id']]
            table['modality'] = 'table'
            table['text'] = construct_full_table_str(table)

            table_list.append(table)
        
        # generate context for baseline method 
        data = {
            "meta": example, 
            "text_content": text_list, 
            "image_content": image_list, 
            "table_content": table_list, 
            "image_captions":image_caption_list,
            "image_embed":image_embed, 
            "triple_embed":triple_embed
        }
        context_str = edit_supporting_context(data)
        context_str = "Question: " + data['meta']['question'] + "\n" + context_str
        data['context_str'] = context_str

        return data

    def __len__(self):
        return len(self.train_info)


class ImageDataset(Dataset):
    def __init__(self, 
        image_meta_path, 
        image_path, 
    ):
        self.image_info = read_jsonlines(image_meta_path)

        # self.image_dict = {doc["id"]: doc for doc in image_info}
    
        self.image_path = image_path


    def __getitem__(self, index):
        # img_info = self.image_dict[index] 
        img_info = self.image_info[index]
        img_info['path'] = os.path.join(self.image_path,img_info['path'])
        img_info['modality'] = 'image'
        return img_info

    

    def __len__(self):
        return len(self.image_info)





class TextDataset(Dataset):
    def __init__(self, 
        text_path
    ):
        self.text_content = read_jsonlines(text_path)


    def __getitem__(self, index):
        text_info = self.text_content[index]
        text_info['modality'] = 'text'
        return text_info

    

    def __len__(self):
        return len(self.text_content)


