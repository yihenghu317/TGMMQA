import torch 
import torch.nn as nn
import json
"""
Construct KG, with entities and relations 
also provide source of each entity 
construct the minimum spanning tree source 
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from .Caption import generate_caption
from .utils import read_jsonlines, process_bag_of_words
from .InfoExtraction import OpenInfoExtraction
import networkx as nx 
import re 
import Levenshtein
import os 

class KGConstruction(nn.Module):
    def __init__(self, openinfo,
                 oie_llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct",
                 temp_data_path = "./",
                 ):
        super().__init__()
        self.oie_llm_model_name = oie_llm_model_name
        self.extract_module = openinfo
        self.temp_data_path = temp_data_path

    def text_process(data):
        pass


    def forward(self, data):
        ###### generate caption for input data 
        data_saved =True
        print("******************** Generate Caption")

        if not data_saved:
          
            image_info = generate_caption(data['image_content'])
            with open(os.path.join(self.temp_data_path, 'temp_image_caption.json'), 'w') as f:    
                json.dump(image_info, f, indent=4)
        else:
            image_info_path = os.path.join(self.temp_data_path, 'temp_image_caption.json')
            f = open(image_info_path)
            image_info = json.load(f)
        

        ###### generate triples


        triple_info = []
        triple_saved = True 

        if not triple_saved:
            ############################ question triples 
            question_triple = self.extract_module.extract_triplets(data['meta']['question'])
            triple_info.append({'id' : data['meta']['qid'], 'triple': question_triple, 'modality':'question'})

            ############################ image triples 
            caption_triples = []
            for i in image_info:
                caption_triple = self.extract_module.extract_triplets(i['caption'])
                print(i['id'],caption_triple)
                caption_triples.append(caption_triple) 
                triple_info.append({
                    'id': i['id'], 'triple':caption_triple, 
                    'modality':'image', 'path':i['path']
                })
            ############################ text triples
            text_triples = []
            for text in data['text_content']:
                text_triple = self.extract_module.extract_triplets(text['text'])
                
                text_triples.append(text_triple)
                triple_info.append({
                    'id':text['id'], 'triple':text_triple, 
                    'modality':'text', 'text':text['text']
                })


            with open(os.path.join(self.temp_data_path, 'data_triples.json'), 'w') as f1:    
                json.dump(triple_info, f1, indent=4)
        else: 
            triple_path = os.path.join(self.temp_data_path, 'data_triples.json')
            f = open(triple_path)
            triple_info = json.load(f)
        

        # content information
        all_content = data['image_content']  +data['text_content'] + data['table_content']
        content_info = {doc['id']:doc for doc in (data['image_content'] + data['text_content'] + data['table_content'])}
        question_entities = []
       
        G = nx.Graph()
        for i, t in enumerate(triple_info):
            if t['modality'] == 'question':
                extracted_triples = self.process_triples(t['triple'])
                for triple in extracted_triples:
                    question_entities.append(triple[0])
                    question_entities.append(triple[2])
                continue
            related_meta_info = content_info[t['id']]
            G = self.add_triples(t['triple'], G, t['id'], t['modality'], related_meta_info['title'] )

        ################################## add nodes and edges to graph based on given info 
        # add tables
        for table in data['table_content']:
            G = self.convert_table_graph(G,table)
            

        node_list = list(G.nodes(data=True))
        for i, i_attr in node_list:
            for j, j_attr in node_list:
                if i == j:
                    continue
                common_words = i_attr['bow'] & j_attr['bow']
                if len(common_words) >= 1:
                    G.add_edge(i,j, relation="relate to")
                
        # find the matching entities 
        related_entities = [] 
        question_entities = set(question_entities)
        print("question entities: ", question_entities)

        for e in question_entities:
            qentities_processed = process_bag_of_words(e)
            # print(qentities_processed)
            for node, attributes in list(G.nodes(data=True)):
                # print(node, attributes,"\n")
                common_words = attributes['bow'] & qentities_processed
                if len(common_words) >= 1:
                    # print(" ---- common word: ",e, node,  attributes['bow'])
                    related_entities.append(node)
        return G


        
        
        
    def process_triples(self, triple_str):
        """
        return: list of triples
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
    
    def add_triples(self, triples, G, source_id, source_modality, title):
        """
        input: string of triples
        """
        triples = self.process_triples(triples)
        entities = set()
        for triple in triples: 
            entities.add(triple[0])
            entities.add(triple[2])
        cur_nodes = list(G.nodes(data=True))
        for e in entities:
            bow = process_bag_of_words(e) | process_bag_of_words(title, splitter = ' ')
            G.add_node(e, source_id = source_id, modality = source_modality, bow=bow, title = title)
        for triple in triples:
            G.add_edge(triple[0], triple[2], relation=triple[1])
        # print(G.nodes(data=False))
        return G

    def convert_table_graph(self,G,table_content):
        source_id = table_content['id']
        source_modality = "table"
        col_name = []
        for col in table_content['table']['header']:
            col_name.append(col['column_name'])
        
        table_header = table_content['title']
        G.add_node(table_header, source_id = source_id, modality = source_modality, bow=process_bag_of_words(table_header, splitter =" "), title = table_header)
        for i,row in enumerate(table_content['table']['table_rows']):
            pre_cols = []
            for j, col in enumerate(row):
                if len(col['text']) == 0:
                    continue
                col_value = f"{col_name[j]}_{ col['text']}"
                G.add_node(col_value, source_id=source_id, modality = source_modality, bow=process_bag_of_words(col['text'], splitter =" "), title=col['text'])
                
                for pc in pre_cols:
                    G.add_edge(pc, col_value, relation="relate to")
                pre_cols.append(col_value)

        return G
    
    def k_hop_subgraph(self,G, nodes, k = 1):
        """
        nodes: list of node name
        """
        nodes = set(nodes)
        length, path = nx.multi_source_dijkstra(G, nodes)
        subgraph_nodes = []
        for node, dist in length.items():
            if dist <= k: 
                subgraph_nodes.append(node)
                

        
        subgraph = G.subgraph(subgraph_nodes)
        # print(subgraph.nodes(data=True))
        sources = nx.get_node_attributes(subgraph, "source_id")
        sources = set(list(sources.values()))

        # return subgraph
        return sources
    
    def add_new_triplets(self, G, new_triples, modality="question"):
        """
        new_triples: string of triples, "[['a','b','c'],[1,2,3]]"
        add new node and relation into graph 
        connects with other nodes that have common words
        """

        triples = self.process_triples(new_triples)
        entities = set()
        for triple in triples: 
            entities.add(triple[0])
            entities.add(triple[2])
        # G = nx.Graph()
        cur_nodes = list(G.nodes(data=True))
        new_nodes =  []
        for e in entities:
            bow = process_bag_of_words(e) 
            new_nodes.append(e)
            
            G.add_node(e, source_id = None, modality = modality, bow=bow, title = e)
            for node, attrs in cur_nodes:
                comm_words = attrs['bow'].intersection(bow)
                if len(comm_words) > 1:
                    G.add_edge(e, node, relation="relate to")
                    G.add_edge(node, e, relation="relate to")
        
        for triple in triples:
            G.add_edge(triple[0], triple[2], relation=triple[1])
        
        return G, new_nodes

    def to_triplet_string(self, G):
        # print(G.edges(data=True))
        triples = []
        for head, tail, attribute in G.edges(data=True):
            triple = [head, attribute['relation'], tail]
            triples.append(triple)
        
        return triples





        







        
        
        


        

        

        


