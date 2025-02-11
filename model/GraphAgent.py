import torch 
import torch.nn as nn
import json
from PIL import Image
import re 

from .utils import extract_question_modality
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
import os 
import bisect

class Agent(nn.Module):
    def __init__(self, reason_module, 
                llm_pipe, 
                vl_model, vl_processor,
                few_shot_path = './',
                log = True,
                pipe_model_id = None
                 ):
        super().__init__()
        self.llm_pipe = llm_pipe
        self.vl_model = vl_model 
        self.vl_processor = vl_processor
        self.reason_module = reason_module
        self.ques_no = 0
        self.evid_no = 0
        self.ans_no = 0
        self.nodes_dict = {}

        # load the prompt to answer the question
        # Multimodal section
        self.few_shot_path = few_shot_path
        self.log = log
        self.pipe_model_id = pipe_model_id

        self.action_prompt = self.load_action_instruction()
    

    def load_action_instruction(self, ): 
        defined_action = ['answer', 'question', 'retrieval', 'stop']
        action_prompt= {}
        for action in defined_action: 
            instruction, examples = self.read_instruction(action = action)
            action_prompt[action] = {}
            action_prompt[action]['instruction'] = instruction
            action_prompt[action]['examples'] = examples
        return action_prompt

        
    


    def read_instruction(self, action = 'retrieval'): 
        text = open(os.path.join(self.few_shot_path, f'actions/{action}_instruction.txt')).read() 

        lines = text.strip().split("\n")
        instruction = ""
        examples = []
        
        # Extract the instruction part
        instruction_start = False
        for i, line in enumerate(lines):
            if line.startswith("Instruction:"):
                instruction = ""
                instruction_start = True
                continue
            if instruction_start and line.strip() == "":
                break
            if instruction_start:
                instruction += " " + line.strip()
        
        # Collect examples by detecting empty lines between them
        current_example = []
        for line in lines[i+1:]:  # Start after the instruction
            line = line.strip()
            if "Example:" in line: 
                continue
            if line == "" and current_example:  # Empty line indicates end of an example
                examples.append("\n".join(current_example).strip())
                current_example = []
            elif line:  # Non-empty line, part of an example
                current_example.append(line)
        
        # Add the last example if any
        if current_example:
            examples.append("\n".join(current_example).strip())
        
        return instruction.strip(), examples


    def construct_action_prompt(self, latest_action): 

        defined_action = {
            
            'Q': 'question', 
            'S': 'stop', 
            'E': 'retrieval', 
            'A': 'answer', 
        }

        instruct_str = ""
        example_str = ""

        ii = 1 
        ei = 1
        for key, paction in defined_action.items(): 
            if key in latest_action: 
                continue 
            prompt = self.action_prompt[paction]
            instruct = prompt['instruction']
            examples = prompt['examples']
            instruct_str += f"{ii}: {instruct}\n"
            ii += 1
            for example in examples:
                example_str += f"Example {ei}:\n{example}\n\n"
                ei += 1
        return instruct_str, example_str


    def run_llm(self,messages, string_length = 1280, given_model_id = None):
        response = self.llm_pipe.run_model(messages, string_length = string_length, given_model_id = given_model_id)
        return response

    def run_vlm(self, question, img_path):
        prompt = f"USER: <image>\n{question}\n ASSISTANT:"
        image = Image.open(img_path)
        inputs = self.vl_processor(text=prompt, images=image, return_tensors="pt")
        generate_ids = self.vl_model.generate(**inputs, max_new_tokens=200)
        generated_text = self.vl_processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()

        generated_text = generated_text.split("ASSISTANT:")[-1].strip()
        return generated_text


    def next_action(self, question, data_summary, plan):
        if self.log: 
            print(f"=================== Agent || next_action | @quesion:={question}@data_summary:={data_summary}@plan:={plan}&===================")
            print("==node dict:", self.nodes_dict)
       
        latest_node = list(self.nodes_dict)[-1]
        action_instruct, action_examples = self.construct_action_prompt(latest_node)

        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/got.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'got.txt')).read()
        filled_prompt = prompt_template.format_map({
            "plan":plan, 
            "sources":data_summary,
            "question":question, 
            "graph_description":self.parse_graph(),
            "node_list":self.full_node_list(),
            "instruction": action_instruct, 
            "examples": action_examples
        })
        messages = [
            {"role": "system", "content": "You are an highly intelligent, goal-driven assistant. Your task is to carefully plan the next steps to answer the given question based on given instructions and examples. Focus on one step at a time, making sure each step is well thought out and moves closer to answering the question."},
            {"role": "user", "content": filled_prompt},
        ]
        print(filled_prompt, "------------\n\n\n")

        action =  self.run_llm(messages,string_length=2000)
        print("\n\n[[[Response: ]]]\n", action, "\n]]]]]]]]]]\n\n\n\n")
        
        result = self.reg_pattern_extact(action)

        return result



    def full_node_list(self,):
        return str(list(self.nodes_dict.keys()))
    
    def context_node_str(self, node_list):
        result = ""
        for node in node_list:
            if node not in self.nodes_dict.keys():
                continue
            node_info = self.nodes_dict[node]
            node_str = node
            node_str += ": " + node_info["text"]
            result += node_str + "\n"
        
        return result
            


    def parse_graph(self,): 
        result = ""
        for node, content in self.nodes_dict.items():
            if node == "Q0":
                continue
            pre_base = ""
            if len(content['parent']) != 0: 
                pre_base += "Based on " + self.parse_node_list(content['parent']) + ", "
                result += pre_base 
            result += f"{self.node_type(node)} {node} is generated: {content['text']}\n\n"

        return result
            
        

    def node_type(self, input_str):
        input_str = input_str.upper()
        if "E" in input_str or 'R' in input_str:
            return "Evidence"
        if "Q" in input_str: 
            return "Question"
        if "A" in input_str:
            return "Answer"
        return ""
    
    def parse_node_list(self, node_list):
        nodes_str = ""
        for i, node in enumerate(node_list):
            if "Q" in node: 
                nodes_str += f"question {node}"
            if "E" in node: 
                nodes_str += f"evidence {node}"
            if "A" in node: 
                nodes_str += f"answer {node}"
            
            if i == len(node_list) - 1: 
                nodes_str += ""
            else: 
                nodes_str += ", "
        
        return nodes_str
            

    def process_node(self, input_str):
        """
        for a string representing list of nodes, extract list of node
        """       
        if input_str is None: 
            return []
        input_str = input_str.replace("'","").replace("[", "").replace("]", "").replace(" ", "")
        node_list = input_str.split(",")
        result_list = []
        for node in node_list:
            pattern = r"^[QE]\d+$"
            is_valid = bool(re.match(pattern, node))
            if is_valid: 
                result_list.append(node)

        return node_list
        



    def add_node(self, node, parent, node_type): 
        """
        add a new node
        generate a node id
        parent should be a list of node
        """
        idx = 0 
        if node_type[0] == "Q": 
            idx = self.ques_no 
            self.ques_no += 1
        if node_type[0] == "E" or node_type[0] == 'R': 
            idx = self.evid_no 
            self.evid_no += 1
        if node_type[0] == "A": 
            idx = self.ans_no
            self.ans_no += 1

        # add id of new node 
        node_id = f"{self.node_type(node_type)[0]}{idx}"
        self.nodes_dict[node_id] = {
            "text": node, 
            "parent": parent
        }
        return 
    
    def add_node_byaction(self, action): 
        """
        add a new node
        generate a node id
        parent should be a list of node
        """
        node = action['node']
        parent = action['based_on']
        node_type = action['node_type'].upper()
        idx = 0 
        
        if node_type[0] == "Q": 
            idx = self.ques_no 
            self.ques_no += 1
        if node_type[0] == "E" or node_type[0] == 'R': 
            idx = self.evid_no 
            self.evid_no += 1
        if node_type[0] == "A": 
            idx = self.ans_no
            self.ans_no += 1

        # add id of new node 
        node_id = f"{self.node_type(node_type)[0]}{idx}"
        self.nodes_dict[node_id] = {
            "text": node, 
            "parent": parent
        }
        return 
    

    def extract_content(self, content):
        if self.log: 
            print(f"=================== Agent || extract_content | @content:={content}&===================")

        
        """
        extract the key instruction from the reasoning sentense generated by LLM
        """

        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/content_extraction.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'content_extraction.txt')).read()
        
        filled_prompt = prompt_template.format_map({
            "context":content, 
        })
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in extracting the right information. "},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)

        result = self.reg_pattern_extact(response)

        return result
    
    def reg_pattern_extact(self, text):
        new_node_match = re.search(r'\[?\s*Node content\s*]?:\s*(.*)', text, re.IGNORECASE)
        based_on_match = re.search(r'\[?\s*Based on\s*]?:\s*(.*)', text, re.IGNORECASE)
        node_type_match = re.search(r'\[?\s*Type of New Node\s*]?:\s*(.*)', text, re.IGNORECASE)
        new_node_content = new_node_match.group(1) if new_node_match else None
        based_on_content = based_on_match.group(1) if based_on_match else None
        node_type_content = node_type_match.group(1) if node_type_match else None

        based_on_content = self.process_node(based_on_content)
        result = {
            "node": new_node_content, 
            "based_on": based_on_content, 
            "node_type": node_type_content
        }
        return result

    def excute_action(self, instruct_dict, G, all_content_dict):
        """
        execute the action given by input
        """
        # if 'q' in instruct_dict["node_type"].lower():
        #     # node type is question 
        #     self.add_node(instruct_dict['node'], [], "Question")
        #     print("done add question node")
    
        
        return 
    

        