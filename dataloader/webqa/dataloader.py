import torch
from torch.utils.data import Dataset
import jsonlines
import pickle
import numpy as np
import os 
import json
from dataloader.utils import edit_supporting_context, construct_full_table_str
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


class WebQATrainDataset(Dataset):
    def __init__(
        self, train_path, line_path, image_tsv_path,  temp_path = './', train = True 
    ):
        with open(train_path, 'r') as file: 
            data = json.load(file)
        self.train_data = data 
        self.train_data_key = list(data.keys())

        self.img_fp = open(image_tsv_path, "r")
        with open(line_path, "r") as fp_lineidx:
            lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        self.lineidx = lineidx
        self.temp_path = temp_path

        fsplit = 'train'
        if not train: 
            fsplit = 'val'
            
        self.train_idx = np.load(os.path.join(
            "/data1/yihenghu/data_discovery/mmkg_vq/dataloader/webqa", 
            f"{fsplit}_idx.npy"
        ))

        

        
    

    def __getitem__(self, index):
        """
        (['Q', 'A', 'topic', 'split', 'Qcate', 'Guid', 'img_posFacts', 'img_negFacts', 'txt_negFacts', 'txt_posFacts']
        
        text: ['title', 'fact', 'url', 'snippet_id']

        image: ['image_id', 'title', 'caption', 'url', 'imgUrl']

        """
        key = self.train_idx[index]
        data = self.train_data[key]
        qid = data['Guid']

        img_content = []
        for img in data['img_posFacts']:
            image_id = img['image_id']
            lidx = self.lineidx[int(image_id) % 10000000]
            img['id'] = str(image_id)
            img['lineidx'] = lidx
            img['modality'] = 'image'
            img_content.append(img)
            
        for img in data['img_negFacts']:
            image_id = img['image_id']
            img['id'] = str(image_id)
            lidx = self.lineidx[int(image_id) % 10000000]
            img['lineidx'] = lidx
            img['modality'] = 'image'
            img_content.append(img)
        
        image_embed = [] 
        if len(img_content) > 0:
            image_embed_path = os.path.join(self.temp_path, f'webqa_image_embed/{qid}.npy')
            if os.path.exists(image_embed_path):
                try:
                    image_embed = np.load(image_embed_path)
                    image_embed = image_embed
                except:
                    print(f"Something wrong with image embed: {qid}")    
        

        
        text_content = data['txt_negFacts'] + data['txt_posFacts']
        for text in text_content: 
            text['id'] = text['snippet_id']
            text['modality'] = 'text'
        
        triple_embed = []
        if len(text_content) > 0:
            triple_embed_path = os.path.join(self.temp_path, f'webqa_triple_embed/{qid}.npy')
            if os.path.exists(triple_embed_path):
                try:
                    triple_embed = np.load(triple_embed_path, allow_pickle=True).item()
                except:
                    print(f"Something wrong with triple embed: {qid} | {triple_embed_path}")



        data['image_content'] = img_content
        data['text_content'] = text_content
        data['image_embed'] = image_embed 
        data['triple_embed'] = triple_embed
        
        return data 

    def __len__(self):
        # return len(self.train_data_key)
        return len(self.train_idx)



class WebQATestDataset(Dataset):
    def __init__(
        self, test_path, line_path, image_tsv_path, 
        temp_path = './'
    ):
        with open(test_path, 'r') as file: 
            data = json.load(file)
        self.test_data = data 
        self.data_key = list(data.keys())

        self.img_fp = open(image_tsv_path, "r")
        with open(line_path, "r") as fp_lineidx:
            lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        self.lineidx = lineidx
        self.temp_path = temp_path

    

    def __getitem__(self, index):
        """
        (['Q', 'A', 'topic', 'split', 'Qcate', 'Guid', 'img_posFacts', 'img_negFacts', 'txt_negFacts', 'txt_posFacts']
        
        text: ['title', 'fact', 'url', 'snippet_id']

        image: ['image_id', 'title', 'caption', 'url', 'imgUrl']

        """
        
        key = self.data_key[index]
        data = self.test_data[key]
        qid = data['Guid']

        img_content = []
        for img in data['img_Facts']:
            image_id = img['image_id']
            lidx = self.lineidx[int(image_id) % 10000000]
            img['ll'] = int(image_id) % 10000000
            img['id'] = str(image_id)
            img['lineidx'] = lidx
            img['modality'] = 'image'
            img_content.append(img)
        
        image_embed = [] 
        if len(img_content) > 0:
            image_embed_path = os.path.join(self.temp_path, f'webqa_image_embed/{qid}.npy')
            if os.path.exists(image_embed_path):
                try:
                    image_embed = np.load(image_embed_path)
                    image_embed = image_embed
                except:
                    print(f"Something wrong with image embed: {qid}")    
        
            
        text_content = data['txt_Facts']
        for text in text_content: 
            text['id'] = text['snippet_id']
            text['modality'] = 'text'

        triple_embed = []
        if len(text_content) > 0:
            triple_embed_path = os.path.join(self.temp_path, f'webqa_triple_embed/{qid}.npy')
            if os.path.exists(triple_embed_path):
                try:
                    triple_embed = np.load(triple_embed_path, allow_pickle=True).item()
                except:
                    print(f"Something wrong with triple embed: {qid} | {triple_embed_path}")


        data['image_content'] = img_content
        data['text_content'] = text_content
        data['image_embed'] = image_embed 
        data['triple_embed'] = triple_embed
        
        return data 

    def __len__(self):
        return len(self.data_key)

