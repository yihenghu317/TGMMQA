from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import torch.nn as nn
import numpy as np
import os 
import bisect

class OpenInfoExtraction(nn.Module):
    def __init__(self, pipeline, 
        oie_few_shots_example_path ='oie_few_shot_examples.txt', 
        oie_prompt_template_path =  'oie_template.txt', 
        question_template_path = 'question_triple_example.txt', 
        few_shot_path = './',
        log = True,
    ):  
        super().__init__()

        self.few_shot_path = few_shot_path
        oie_few_shots_example_path = os.path.join(self.few_shot_path, oie_few_shots_example_path)
        oie_prompt_template_path = os.path.join(self.few_shot_path, oie_prompt_template_path)
        question_template_path= os.path.join(self.few_shot_path, question_template_path)


        self.llm_pipe = pipeline 
        self.oie_few_shots_example = open(oie_few_shots_example_path).read()
        self.oie_prompt_template = open(oie_prompt_template_path).read()

        self.question_template = open(question_template_path).read()
        self.log = log

            



    
    def forward(self, ):
        pass 
    
    def run_llm(self,messages, string_length = 1280, given_model_id = None):
        response = self.llm_pipe.run_model(messages, string_length = string_length, given_model_id = given_model_id)
        return response

        

    def extract_triplets(self, text, string_length = 1280):
        if self.log: 
            print(f"=================== OpenInfoExtraction || extract_triplets | @text:={text}@string_length:={string_length}&===================")
        filled_prompt = self.oie_prompt_template.format_map({
            "few_shot_examples": self.oie_few_shots_example,
            "input_text": text,
        })


        messages = [
            {"role": "system", "content": "You are a assistant to generate content strictly following given examples."},
            {"role": "user", "content": filled_prompt},
        ]
        assistant_response = self.run_llm(messages, given_model_id = 'llama3-8b-instruct')
        return assistant_response
    
    def extract_question_triple(self, text):
        if self.log: 
            print(f"=================== OpenInfoExtraction || extract_question_triple | @text:={text}&===================")
        # filled_prompt = self.question_template.format_map({
        #     "question": text,
        # })
        # question_key_triple
        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/question_key_triple.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'question_keyword.txt')).read()


        filled_prompt = prompt_template.format_map({
            "question": text,
        })

        messages = [
            {"role": "system", "content": "You are a assistant to generate content strictly following given examples."},
            {"role": "user", "content": filled_prompt},
        ]
        response = self.run_llm(messages, given_model_id = 'llama3-70b-instruct')

        return response





