import torch 
import torch.nn as nn
import json
import re 

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
# from .KgConstruction import KGConstruction
from .GraphConstruction import KGConstruction
from .ReasonModule import ReasonModule
from .GraphAgent import Agent
from .InfoExtraction import OpenInfoExtraction
from .FoundationModels.llama_model import LlamaApi
from .FoundationModels.local_model import TextGenerationModel
from .FoundationModels.llava_localmodel import LlavaLocalModel
from .FoundationModels.clip_localmodel import ClipLocalModel
from .FoundationModels.blip_replicate import BLIPReplicateApi

from transformers import pipeline
import torch 
import networkx as nx
from transformers import AltCLIPModel, AltCLIPProcessor, AutoProcessor
import os 
import openai
from openai import OpenAI


class GraphFramework(nn.Module):
    def __init__(self, 
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct", 
        visual_modal_id = "llava-hf/llava-1.5-13b-hf", 
        temp_data_path = "./",
        few_shot_path = "./",
        half_prec = True, 
        use_api = 'llama', 
        debug = False,
        llava_api = True,
        mm_model = 'llava',
    ):
        super().__init__()
        self.model_id = model_id 

        model_kwargs = {}
        if half_prec: 
            model_kwargs = {"torch_dtype": torch.bfloat16}

        use_api = use_api.lower()
        if 'llama' in use_api: 
        else: 
            self.pipe =TextGenerationModel(self.model_id)

    
        self.temp_data_path = temp_data_path
        self.few_shot_path = few_shot_path
        
        
        if debug: 
            # load clip 
            self.md_model, self.md_processor = None, None
            self.visual_model, self.visual_processor = None, None
        else:
            mm_model = mm_model.lower()
            if 'blip' in mm_model:
                self.visual_model = BLIPReplicateApi('blip2')
            else:
                if llava_api:
                    self.visual_model = LlavaReplicateApi('llava-13b')
                else:
                    self.visual_model = LlavaLocalModel()
            self.md_model = ClipLocalModel()
            
        self.reasonmodule = ReasonModule(self.pipe, self.visual_model,  few_shot_path = self.few_shot_path) 
        self.infomodule = OpenInfoExtraction(self.pipe, few_shot_path=self.few_shot_path, )
        


    def forward(self, data):
        """
        Iteratively solve multihop question
        """
        # initialize new agent 
        graphmodule = KGConstruction(self.infomodule, self.md_model,temp_data_path = self.temp_data_path, few_shot_path = self.few_shot_path)
        agent  = Agent(self.reasonmodule , self.pipe, None, None, few_shot_path = self.few_shot_path, )
        data_summary = self.data_content_summary(data)
        

        all_content = data['image_content']  +data['text_content'] + data['table_content']
        all_content_dict = {doc["id"]:doc for doc in all_content}
        all_content_title = {doc["title"]:doc for doc in all_content}


        evidence = []
        G = graphmodule(data)
        question = data['meta']['question']
        plan1 = self.reasonmodule.chain_of_thought(question)
        plans =[plan1]
        plan = ""
        for i, p in enumerate(plans): 
            plan += f"\n***************plan example {i} *******************\n" + p 



        agent.add_node(question, [], "Q")
        total_turn = 15
        t = 0
        max_turn = 5
        i = 0
        while i < max_turn and t < total_turn:
            t += 1
            action = agent.next_action(question, data_summary, plan)
            
            if action["node_type"] is None:
                continue
            if len(str(action['node_type'])) ==0:
                continue
                
            if 'stop' in action["node_type"].lower():
                break

            execute = self.execute_action(agent,graphmodule , action, G, all_content_dict, question)
            if execute is False:
                print("---Missed")
                continue
            
            if "retri" in action["node_type"].lower():
                evidence += execute
                i += 1
            
            torch.cuda.empty_cache()
        
        reason_path = agent.parse_graph()
        
        final_answer = self.reasonmodule.final_answer_generation(question, reason_path, relax = True)
        

        ans = self.reasonmodule.answer_extraction(question, final_answer)
        return ans
        


    def execute_action(self, agent,graphmodule , action_dict, G, all_content_dict, orig_ques, ):
        context = agent.context_node_str(action_dict['based_on'])
        if "retri" in action_dict["node_type"].lower(): 
            retrieval_list, source_list = self.retrieval_type(graphmodule , action_dict['node'], context, G, all_content_dict, orig_ques, modality="text")
            if retrieval_list is None: 
                return False
            for evidence in retrieval_list: 
                agent.add_node(evidence, action_dict['based_on'], "E")
            return source_list
        
        elif "ques" in action_dict["node_type"].lower():
            agent.add_node(action_dict["node"], action_dict['based_on'], "Q")

        elif "ans" in action_dict["node_type"].lower():
            result = self.reasonmodule.final_answer_generation(action_dict['node'], context)
            agent.add_node(result, action_dict['based_on'], "A")

        else: 
            return False
            
        return True
    
        
        
    def parse_evidence(self):
        return "\n".join(self.evidence)
    
    def data_content_summary(self, data):
        result = "images about "
        for img in data['image_content']:
            result += img['title'] + ","
        
        result += " documents about "
        for doc in data['text_content']:
            result += doc['title'] + ","

        result += " and table about "
        for table in data['table_content']:
            result += table['title']
        result += "."
        return result
        

    def retrieval_type(self,graphmodule, subquestion, context, G, all_content_dict, orig_ques, modality="text"):
        
        modality_response = self.reasonmodule.find_modality(subquestion)
        print("**Find Modality: ", modality_response)
        modality = None
        if 'image' in modality_response.lower() or 'visual' in modality_response.lower(): 
            modality = 'image'
        elif 'both' in modality_response.lower():
            modality = 'both'
        else: 
            modality = 'text'
        #############################################
        

        # img_retrieval_instruction  = self.reasonmodule.retrieval_instruction(subquestion, context, modality='image')
        img_retrieval_instruction  = self.reasonmodule.retrieval_instruction(subquestion, context, modality=modality)
        selection = img_retrieval_instruction['selection']
        # check if selection is empty
        s_str = ''.join(selection)
        if len(s_str) == 0:
            selection = []
            img_retrieval_instruction['selection'] = []

        transformed_ques = img_retrieval_instruction['question']
        print("**Retrieval Instruction: ", img_retrieval_instruction)


        # sources: list
        sources = []
        if len(selection) == 0 or modality == 'text':
            subq_triplet = self.infomodule.extract_question_triple(subquestion) 
            print("\n**sub triplets: ", subq_triplet, "\n\n")
            graph, new_nodes = graphmodule.add_new_triplets(G, subq_triplet)
            sources += graphmodule.k_hop_subgraph(graph, new_nodes, modality = "text")


            if transformed_ques is not None and transformed_ques != '' and len(transformed_ques) != 0 and transformed_ques.lower() != 'none':
                subq_triplet = self.infomodule.extract_question_triple(transformed_ques) 
                print("\n** second sub triplets:", subq_triplet, "\n\n")
                if len(subq_triplet) != 0 and subq_triplet != '[]' :
                    graph, new_nodes = graphmodule.add_new_triplets(G, subq_triplet)
                    sources += graphmodule.k_hop_subgraph(graph, new_nodes, modality = "image")
        if len(selection) != 0: 
            # TODO: find new nodes based on selection 
            for select in selection:
                print("selecting: ", select)
                graph, new_nodes = graphmodule.add_new_triplets(G, [str(select)])
                new_sources = graphmodule.nearest_neighbor_selection(graph, new_nodes, modality = 'image')
                sources += new_sources

        # eliminate duplicate: 
        sources = list(set(sources))

        info_list = []
        for i, s in enumerate(sources): 
            if s is None: 
                continue
            info = all_content_dict[s]
            print("\n**Retrieval Examine: ", info['title'])
            if info['modality'] == 'image':
                result = self.reasonmodule.process_source_data(transformed_ques, info, orig_ques)           
            else: 
                result = self.reasonmodule.process_source_data(subquestion, info, orig_ques)      
            print("**Single retrieval result: ", result, "\n\n")     
            info_list.append([info, result])

        if len(info_list) == 0: 
            return None,[]

        retrieve_result = []
        positive_evd = ""
        for info, result in info_list: 
            positive_evd += f"""* Retrieval result from the {info['modality']} of title {info['title']}: \n"{result}". \n\n\n"""
        result = self.reasonmodule.retrieval_extraction_all(subquestion, positive_evd,  orig_ques = orig_ques)
        print("\n**retrieval all result: ", result, "\n\n")
        if "no result" in result.lower():
            return [f"""No information can be found according to the instruction "{subquestion}". Please relax the contraint for retrieval."""], []
        else:
            split_string = result.split("Title of sources")

            # First part is the result section
            result_section = split_string[0].strip()
            sources_section = "Title of sources" + split_string[1].strip()

            result = result_section

            strings_list = re.findall(r'\[([^\]]+)\]', sources_section)

            sources_list = [item.strip() for item in strings_list[0].split(',')]

        result = f"""Based on instruction {subquestion}, we found:{result}"""    
        retrieve_result.append(result)
        return retrieve_result, sources_list
                

            
        


            
        

