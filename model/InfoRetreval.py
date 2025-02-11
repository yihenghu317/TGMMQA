import torch 
import torch.nn as nn
import json
import os 

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
"""

Serve as a connection to graph module and retreve relevant information 
include validation for multimodal information 
"""

class RetrievalModule(nn.Module):
    def __init__(self,
                llm_pipe, 
                vl_model, vl_processor,
                few_shot_path = './'
                 ):
        super().__init__()
        self.llm_pipe = llm_pipe
        self.vl_model = vl_model 
        self.vl_processor = vl_processor
        self.few_shot_path = few_shot_path


        # load the prompt to answer the question
        # Multimodal section

    def run_llm(self,messages):
        terminators = [
            self.llm_pipe.tokenizer.eos_token_id,
            self.llm_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.llm_pipe(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        assistant_response = outputs[0]["generated_text"][-1]["content"]
        return assistant_response

    
    def extract_subquestion(self, question, evidence):
        # generate next subquestion to extract the information we need
        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/reasoning.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'reasoning.txt')).read()
        filled_prompt = prompt_template.format_map({
            "evidence":evidence, 
            "question":question,
        })
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in choosing the right answer to the question. Note that insufficient information to answer questions is common, because you do not have any information about the picture or context."},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)


        # prompt_template_extract = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/question_extraction.txt').read()
        prompt_template_extract = open(os.path.join(self.few_shot_path, 'question_extraction.txt')).read()
        filled_prompt = prompt_template_extract.format_map({
            "context":response, 
        })
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in choosing the right answer to the question. Note that insufficient information to answer questions is common, because you do not have any information about the picture or context."},
            {"role": "user", "content": filled_prompt},
        ]
        subquestion =  self.run_llm(messages)
        return subquestion

        
    
    
