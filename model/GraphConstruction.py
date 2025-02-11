import torch 
import torch.nn as nn
import json
from PIL import Image
import requests
"""
Construct KG, with entities and relations 
also provide source of each entity 
construct the minimum spanning tree source 
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from .Caption import construct_generate_caption
from .utils import read_jsonlines, process_bag_of_words, to_numpy, string_to_list
from .InfoExtraction import OpenInfoExtraction
import networkx as nx 
import re 
from sklearn.neighbors import NearestNeighbors
import Levenshtein
import os 


class KGConstruction(nn.Module):
    def __init__(self, infomodule,
                 multimodal_model, 
                 oie_llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
                 temp_data_path = "./",
                 few_shot_path = "./",
                 pipe_model_id = None,
                 ):
        super().__init__()
        self.temp_data_path = temp_data_path
        self.few_shot_path = few_shot_path

        self.multimodal_model = multimodal_model 
        self.extract_module = infomodule
    
    def node_representation(self, input_text = None, img_path = None ):
        """
        convert single input into representation
        input should be one of the input_text or img_path
        output size: [768]
        """
    

        if input_text is not None:
            text_feature = self.multimodal_model.run_model(text = input_text)
            text_feature = to_numpy(text_feature)[0]

            return text_feature
        
        if img_path is not None:
            image = Image.open(img_path)
            image_features = self.multimodal_model.run_model(image = image)
            image_features = to_numpy(image_features)[0]

            return image_features

        return 
    

    def batch_node_representation(self, input_text_list = None, img_path_list = None):
        """
        input_text_list: ["A", "B", "C"]
        """
        if input_text_list is not None:
            text_feature = self.multimodal_model.run_model(text = input_text_list)
            text_feature = to_numpy(text_feature)

            return list(text_feature)
        
        if img_path_list is not None:
            img_list = []
            for img_path in img_path_list:
                image_file = Image.open(img_path)
                img_list.append(image_file)
            if len(img_list) == 0: 
                return []
            
            image_features = self.multimodal_model.run_model(image = img_list)
            image_features = to_numpy(image_features)

            return list(image_features)
        return []


    

    def forward(self, data):
        ###### generate caption for input data 
        data_saved = False
        image_info = data['image_captions']


        ###### generate triples
        triple_info = []
        triple_saved = False

        file_path = os.path.join(self.temp_data_path, f'multimodal_data_triple/{data["meta"]["qid"]}_data_triples.json')
        if os.path.exists(file_path):
            triple_saved = True


        if not triple_saved:
            ############################ question triples 
            question_triple = self.extract_module.extract_triplets(data['meta']['question'])
            triple_info.append({'id' : data['meta']['qid'], 'triple': question_triple, 'modality':'question'})

            ############################ image triples 
            caption_triples = []
            for i in image_info:
                caption_triple = self.extract_module.extract_triplets(i['caption'])
                caption_triples.append(caption_triple) 
                triple_info.append({
                    'id': i['id'], 'triple':caption_triple, 
                    'modality':'image', 'path':i['path']
                })
            ############################ text triples
            text_triples = []
            for text in data['text_content']:
                text_triple = self.extract_module.extract_triplets(text['text'],string_length = len(text['text'])*2)
                
                text_triples.append(text_triple)
                triple_info.append({
                    'id':text['id'], 'triple':text_triple, 
                    'modality':'text', 'text':text['text']
                })
            

            with open(os.path.join(self.temp_data_path, f'multimodal_data_triple/{data["meta"]["qid"]}_data_triples.json'), 'w') as f1:
                json.dump(triple_info, f1, indent=4)
        else: 
            triple_path = os.path.join(self.temp_data_path, f'multimodal_data_triple/{data["meta"]["qid"]}_data_triples.json')
            f = open(triple_path)
            triple_info = json.load(f)
        

        # content information
        all_content = data['image_content']  +data['text_content'] + data['table_content']
        self.all_content_dict = {doc["id"]: doc for doc in all_content}
    
        content_info = {doc['id']:doc for doc in (data['image_content'] + data['text_content'] + data['table_content'])}
        question_entities = []
       
        # add triple to the graph 
        ################################## add nodes and edges to graph based on given info 
        self.image_available = False
        if len(data['image_content']) > 0:
            self.image_available = True

        self.text_available = False
        if len(data['text_content']) > 0 or len(data['table_content']) > 0:
            self.text_available = True  

        triple_embed_path = os.path.join(self.temp_data_path, f'multimodal_triple_embed/{data["meta"]["qid"]}.npy')
        triple_embed_saved = False

        if len(data['triple_embed']) != 0:
            triple_embed_saved = True


        if not triple_embed_saved:
            # add triples from text and image captions
            self.node_list = []
            self.embed_list = []
            self.meta_info_node = {}
            text_list = []

            
        
            # add triples from triple_info 
            for i, t in enumerate(triple_info):
                if t['modality'] == 'question':
                    continue
                # related_meta_info = content_info[t['id']]
                tps = self.process_triples(t['triple'])
                for tp in tps: 
                    node = str(tp)
                    text_list.append(node)
                    self.node_list.append(t['id'])
                
                # self.add_triples(t['triple'], t['id'], t['modality'], related_meta_info['title'] )
            embeds = self.batch_node_representation(input_text_list = text_list)
            self.embed_list += embeds            
        
            # add triplets of titles
            text_list = []
            for content in all_content: 
                info = "title: " + content['title'] 
                text_list.append(info)
                self.node_list.append(content['id'])
            embeds = self.batch_node_representation(input_text_list=text_list)
            self.embed_list += embeds
            

            # add tables 
            text_list = []
            for table in data['table_content']:
                table_id = table['id']
                # table_rows = self.construct_table_str(table)
                table_elements = self.collect_table_cell(table)
                for row in table_elements:
                    # embed = self.node_representation(input_text=row)
                    text_list.append(row)
                    self.node_list.append(table['id'])
            embeds = self.batch_node_representation(input_text_list=text_list)
            self.embed_list += embeds

            ###################### add images
            self.img_title_node_list = []
            self.img_title_embed_list = []
            img_title_list = []

            if self.image_available:
                for img in data['image_content']:
                    img_title_list.append(img['title'])
                    self.img_title_node_list.append(img['id'])

                it_embeds = self.batch_node_representation(input_text_list=img_title_list)
                self.img_title_embed_list += it_embeds


            ###################### store all the required info 
            node_list_np = np.array(self.node_list,dtype=str)
            title_node_list_np = np.array(self.img_title_node_list, dtype=str)
            store_data = {
                'node_list' : node_list_np, 
                "image_title_node_list":title_node_list_np, 
                "node_embed": np.array(self.embed_list), 
                "image_title_embed": np.array(self.img_title_embed_list)
            }
            np.save(triple_embed_path,store_data)
            print("save triple embed to ",triple_embed_path)
            
        else:
            loaded_data = data['triple_embed']
            self.node_list = loaded_data['node_list']
            self.img_title_node_list = loaded_data['image_title_node_list']
            self.embed_list = loaded_data['node_embed']
            self.img_title_embed_list = loaded_data['image_title_embed']
            print("load data from ",triple_embed_path)

        
        neigh = NearestNeighbors(n_neighbors=5, radius=0.4)
        neigh.fit(self.embed_list)
        self.neigh = neigh


        if len(data['image_content']) > 0:

            self.image_title_graph = NearestNeighbors(n_neighbors=5, radius=0.4)
            self.image_title_graph.fit(self.img_title_embed_list)
            self.image_graph = NearestNeighbors(n_neighbors=5, radius=0.4)
            self.img_embed_list = []
            self.img_node_list = []
            img_path_list = []
            for img in data['image_content']:
                img_path_list.append(img['path'])
                # self.img_embed_list.append(embed)
                self.img_node_list.append(img['id'])

            img_feature_path = os.path.join(self.temp_data_path, f'multimodal_image_embed/{data["meta"]["qid"]}.npy')
            if len(data['image_embed']) != 0:
                embeds = data['image_embed'].tolist()
            else:
                embeds = self.batch_node_representation(img_path_list=img_path_list)
            self.img_embed_list += embeds
            self.image_graph.fit(self.img_embed_list)
            print("  img embed list: ", len(self.img_embed_list))
            print(" img node list: ", len(self.img_node_list))  

        return neigh

        
        
        
    def process_triples(self, triple_str):
        """
        return: list of triples(list)
        """
        triple_str = triple_str.replace(' ', '').replace('\n', '')
        triple_str = triple_str[1:-1]
        pattern = re.compile(r'\[(.*?)\]')

        # Find all matches
        matches = pattern.findall(triple_str)

        # Process matches to extract individual lists
        extracted_lists = []
        for match in matches:
            # Split the items by comma and strip extra whitespace and quotes
            items = [item.strip().strip("'\"") for item in match.split(',')]
            extracted_lists.append(items)
        return extracted_lists
    
    def add_triples(self, triples, source_id, source_modality, title):
        """
        input: given string of a set of triples 
        """
        # generate embedding for each node and store the meta data info
        triples = self.process_triples(triples)
        
        node_embeds = []

        text_list = []
        for triple in triples: 
            
            node = str(triple)
            text_list.append(node)
        
        node_embeds = self.batch_node_representation(input_text_list=text_list)
        return triplets, node_embeds

    def construct_table_str(self, table_data):
        col_name = []
        for col in table_data['table']['header']:
            col_name.append(col['column_name'])

        table_header = f"In table of {table_data['title']}, "
        # table_str = ""
        table_element = []
        for i,row in enumerate(table_data['table']['table_rows']):
            row_str = table_header
            for j, col in enumerate(row): 
                if len(col['text']) == 0: 
                    continue
                row_str += f"{col_name[j]} is {col['text']}"
                if j < len(row) - 1: 
                    row_str += ", "
                else:
                    row_str += "."

            table_element.append(row_str)
        return table_element
    
    def collect_table_cell(self, table_data ):
        col_name = []
        for col in table_data['table']['header']:
            col_name.append(col['column_name'])

        table_element = []
        for i,row in enumerate(table_data['table']['table_rows']):
            for j, col in enumerate(row): 
                if len(col['text']) == 0: 
                    continue
                cell_info = f"{col_name[j]} : {col['text']}"
            
                table_element.append(cell_info)
        return table_element
        
    
    def get_raduis_neighbors(self, graph, graph_node_list, nodes,radius = 20):
        """
        nodes: list of node with 
        """
        dist, nodes_idx = graph.radius_neighbors(nodes , radius, return_distance = True)
        sources = set()
        source_dist = dict()
        # iterate dist and nodes idx 
        for node_dist, node_idx in zip(dist, nodes_idx):
            for d, idx in zip(node_dist, node_idx):
                node =  graph_node_list[idx]
                sources.add(node)
                if node in source_dist.keys():
                    source_dist[node].append(d)
                else:
                    source_dist[node] = [d]
        
        # calculate the distance by average 
        node_avg_dist = {}
        for node, dist_list in source_dist.items():
            node_avg_dist[node] = np.min(dist_list)
        
        node_avg_dist = {k: v for k, v in sorted(node_avg_dist.items(), key=lambda x: x[1])}
        node_avg_dist = {k:v for k,v in node_avg_dist.items() if v < radius}
   
        return list(node_avg_dist.keys())
    
    def k_hop_subgraph(self,graph, nodes, k = 1, radius = 23, modality = 'text'):
        """
        nodes: should be a list of nodes
        return set of source id 
        """
                
        source_ids = []
        if modality == 'text' and self.text_available:
            source_ids = self.get_raduis_neighbors(self.neigh, self.node_list ,nodes, radius=13.5) 
        elif self.image_available:  
            source_by_image = self.get_raduis_neighbors(self.image_graph, self.img_node_list,nodes, radius=23)
            source_ids = source_by_image
        return source_ids
        
        # return subgraph
    
    def k_nearest_neighbor(self, graph,graph_node_list, nodes, k = 1):

        # nodes: list of node embedding, n x d, e.g, 10 x 256
        nodes_idx = graph.kneighbors(nodes, k, return_distance=False)
        # nodes_idx: 2d array, n x k 
        sources = []
        for k_nodes_idx in nodes_idx:
            
            node = [graph_node_list[i] for i in k_nodes_idx]
            # flatten list
            sources += node 

        return sources
        
    
    
    def nearest_neighbor_selection(self, graph, nodes, k= 1, modality = "text"):
        """
        find the nearest neighbor for a list of nodes
        text: search in neigh 
        image: search in img_title
        """
        sources_ids = []
        if modality == "text" and self.text_available:
            sources_ids = self.k_nearest_neighbor(self.neigh, self.node_list ,nodes) 
        elif self.image_available: 
            
            sources_ids = self.k_nearest_neighbor(self.image_title_graph, self.img_title_node_list, nodes,)
        return sources_ids

    
    def add_new_triplets(self, G, triples, modality="question"):
        """
        input triplets: extracted already 
        """

        if not isinstance(triples, list):
            if '[' in str(triples) and ']' in str(triples):
                try: 
                    triples = string_to_list(triples)
                except (ValueError, SyntaxError, TypeError) as e:
                    triples = [str(triples)]
            else:
                triples = [str(triples)]

        node_embeds = []
        for triple in triples: 
            node = str(triple)
            embed = self.node_representation(input_text=node)
            # self.embed_list.append(embed)
            node_embeds.append(embed)
            # self.node_list.append(node)
        
        return G, node_embeds

        

    def to_triplet_string(self, G):
        triples = []
        for head, tail, attribute in G.edges(data=True):
            triple = [head, attribute['relation'], tail]
            triples.append(triple)
        return triples


