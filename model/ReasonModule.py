import torch 
import torch.nn as nn
import json
from PIL import Image
import re 
import bisect

from .utils import extract_question_modality, image_reader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
import os 
import random
import openai
import numpy as np


class ReasonModule(nn.Module):
    def __init__(self,
                llm_pipe, 
                vl_model,
                few_shot_path = './', 
                log = True, 
                img_fp = None, 
                pipe_model_id = None
                 ):
        super().__init__()
        self.llm_pipe = llm_pipe
        self.vl_model = vl_model 
        self.few_shot_path = few_shot_path
        self.log = log 
        self.img_fp = img_fp
        self.pipe_model_id = pipe_model_id
        # with open('/home/hu052/data_discovery/temp_data/webqa_cate.json', 'r') as file: 
        #     cate_ques = json.load(file)
        # self.cate_ques = cate_ques


        # load the prompt to answer the question
        # Multimodal section

    def run_llm_(self,messages, string_length = 1280, given_model_id = None):
        new_token_list = [ 256,  512,  768, 1024, 1280]
        token_len = int(string_length / 5) + 400
        idx = bisect.bisect_right(new_token_list, token_len)
        if idx > len(new_token_list): 
            idx = len(new_token_list) -1 
        new_token_num = new_token_list[idx]

        if given_model_id is None: 
            model_id = self.pipe_model_id
        else: 
            model_id = given_model_id
        try: 
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
                pad_token_id=self.llm_pipe.tokenizer.eos_token_id
            )
            assistant_response = outputs[0]["generated_text"][-1]["content"]
            return assistant_response
        except openai.BadRequestError as e:
            # Handle BadRequestError
            print(f"BadRequestError: {e}")
            return None  # Indicate failure

        except Exception as e:
            print(f"An error occurred: {e}")
            return None # Indicate failure

    def run_llm(self,messages, string_length = 1280, given_model_id = None):
        response = self.llm_pipe.run_model(messages, string_length = string_length, given_model_id = given_model_id)
        return response


    def run_vlm(self, question, img_path, img_lineidx=None):
        if img_lineidx is None:
            generated_text =self.vl_model.run_model(question, img_path)
        else: 
            generated_text = self.vl_model.run_model(question,None, img_fp=self.img_fp, img_lineidx = img_lineidx)
        return generated_text
    
    def extract_subquestion(self, question, evidence):
        if self.log: 
            print(f"=================== ReasonModule || extract_subquestion | @question:={question}@evidence:={evidence}&===================")
        # generate next subquestion to extract the information we need
        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/reasoning.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'reasoning.txt')).read()
        filled_prompt = prompt_template.format_map({
            "evidence":evidence, 
            "question":question,
        })
        # print(f"+++++++++\n{filled_prompt}\n+++++++++\n")
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in finding the correct answer to the question. Note that insufficient context to answer questions is common, because you do not have any information about the external context. "},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)

        subquestion, modality = self.extract_keycontent(response)
        return subquestion,modality
    
    def generate_subquestion(self, question, evidence):
        if self.log: 
            print(f"=================== ReasonModule || generate_subquestion | @question:={question}@evidence:={evidence}&===================")
        prompt_template = open(os.path.join(self.few_shot_path, 'question_reasoning.txt')).read()
        filled_prompt = prompt_template.format_map({
            "evidence":evidence, 
            "question":question,
        })
        
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in finding the correct answer to the question. Note that insufficient context to answer questions is common, because you do not have any information about the external context. "},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)

        subquestion, modality = self.extract_keycontent(response)
        return subquestion,modality
    
    def extract_after_keyword(self, input_string, keyword):
        # Find the index of the specified keyword
        index = input_string.lower().rfind(keyword.lower())
        
        # If the keyword is found, extract everything after it
        if index != -1:
            return input_string[index + len(keyword) :].strip()  # strip() to remove leading whitespace
        else:
            return None


    def generate_answer_action(self, node_summary, instruction):
        if self.log:
            print(f"=================== ReasonModule || generate_answer_action | @node_summary:={node_summary}@instruction:={instruction}&===================")
        
        prompt_template = open(os.path.join(self.few_shot_path, 'answer_action.txt')).read()
        filled_prompt = prompt_template.format_map({
            "nodes":node_summary, 
            "instruction": instruction
        })
        
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in generating the correct response."},
            {"role": "user", "content": filled_prompt},
        ]
        model_id = None

        response =  self.run_llm(messages, string_length = 1000, given_model_id = model_id)
        answer = self.extract_after_keyword(response, "Rephrase:")
        if answer is None: 
            return response
        return answer
        


    def text_reasoning(self, retrieve_info, text, title, orig_ques):
        if self.log: 
            print(f"=================== ReasonModule || text_reasoning | @retrieve_info:={retrieve_info}@text:={text}&===================")
        prompt_template = open(os.path.join(self.few_shot_path, 'text_reasoning.txt')).read()
        filled_prompt = prompt_template.format_map({
            "text":text, 
            "title": title,
            "description": retrieve_info,
            "question":orig_ques
        })
        
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in producing the correct response. "},
            {"role": "user", "content": filled_prompt},
        ]
        model_id = 'llama3-8b-instruct'
        response =  self.run_llm(messages, string_length = len(text), given_model_id = model_id)
        return response

    def image_reasoning(self, question, path, img_lineidx = None):
        if self.log: 
            print(f"=================== ReasonModule || image_reasoning | @question:={question}@path:={path}&===================")
    
        
        response = self.run_vlm(question, path, img_lineidx=img_lineidx)
        
        return response

    def process_source_data(self,question, data, orig_ques, ):
        modality = 'image' if 'path' in data else 'text'
        result = ""
        img_ques = question
        if 'path' in data.keys(): 
            question = img_ques
            result = self.image_reasoning(question, data['path'])
        if 'lineidx' in data.keys():
            question = img_ques
            result = self.image_reasoning(question, "", img_lineidx=data['lineidx'])

        
        if 'text' in data.keys(): 
            result = self.text_reasoning(question, data['text'], data['title'], orig_ques)
        if 'fact' in data.keys():
            result = self.text_reasoning(question, data['fact'], data['title'], orig_ques)
            
        return result

    def extract_string(self, text, keyword):
        pattern = rf"{keyword}:\s*(.*?)(?=(?:\n|$))"
        match = re.search(pattern, text, re.DOTALL)
        
        # Return the extracted string or None if not found
        return match.group(1).strip() if match else None
    
    def text_retrieval_instruction(self, question , context):
        if self.log: 
            print(f"=================== ReasonModule || text_retrieval_instruction | @question:={question}@context:={context}&===================")
        prompt_template = open(os.path.join(self.few_shot_path, 'text_retrieval_instruction.txt')).read()
        filled_prompt = prompt_template.format_map({
            "instruction": question,
            "context": context, 
        })
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to generate transformed text given input. "},
            {"role": "user", "content": filled_prompt},
        ]

        response =  self.run_llm(messages)
        return response
        
        

    def retrieval_instruction(self, question, context, modality = "text"):
        if self.log: 
            print(f"=================== ReasonModule || retrieval_instruction | @question:={question}@context:={context}&===================")
        prompt_template = open(os.path.join(self.few_shot_path, 'retrieval_type.txt')).read()
        filled_prompt = prompt_template.format_map({
            "instruction": question,
            "context": context, 
        })
        messages = [
            {"role": "system", "content": "You are a highly intelligent assistant tasked with analyzing the provided context and instruction to extract and transform the image-related request into a clear, actionable guide for another assistant."},
            {"role": "user", "content": filled_prompt},
        ]
        

        response =  self.run_llm(messages)
        type_ = self.extract_string(response, "Type")
        selection = self.extract_string(response, "Selection")
        question = self.extract_string(response, "Question")
        match = re.search(r'\*\*\s*\[(.*?)\]', selection)
        if match:
            processed_selection = match.group(1)
        else:
            processed_selection = selection.replace('[', '').replace(']', '')
        processed_selection = processed_selection.split(',')
        result = {
            "type": type_, 
            "selection": processed_selection, 
            "question": question
        }
        return result
        
        
    def question_transformation(self,question, modality ="text"):
        if self.log: 
            print(f"=================== ReasonModule || question_transformation | @question:={question}@modality:={modality}&===================")
        """
        Generate the subquestion that could be fed into other multimodal module
        """
        if modality == "text": 
            return {
                "question": question, 
                 "expected_answer": "",
            }
        # prompt_template = open("/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/question_transform.txt").read()
        prompt_template = open(os.path.join(self.few_shot_path, 'question_transform.txt')).read()
        filled_prompt = prompt_template.format_map({
            "input_question": question
        })
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to generate transformed text given input. "},
            {"role": "user", "content": filled_prompt},
        ]

        text =  self.run_llm(messages)
        # result = output.split(':')[-1].strip()
        result = {}

        # Extract "Transform"
        transformed_q = self.extract_string(text, "Transformed Question")
        expected_a = self.extract_string(text, "Expected Answer")
        if 'yes' in expected_a.lower():
            expected_a = 'yes'
        elif 'no' in expected_a.lower():
            expected_a = 'no'
        else:
            expected_a = ""    
            
        result = {
            "question": transformed_q, 
            "expected_answer": expected_a
        }
        return result
    



    def extract_keycontent(self, input_text_str):
        if self.log: 
            print(f"=================== ReasonModule || extract_keycontent | @input_text_str:={input_text_str}&===================")
        prompt_template = open(os.path.join(self.few_shot_path, 'question_extraction.txt')).read()
        filled_prompt = prompt_template.format_map({
            "context":input_text_str, 
        })
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in generating the right response to the question. "},
            {"role": "user", "content": filled_prompt},
        ]
        subquestion =  self.run_llm(messages)
        question, modality = extract_question_modality(subquestion)


        return question, modality

    
    def answer_examination(self, subques, ans, orig_ques, title):
        if self.log: 
            print(f"=================== ReasonModule || answer_examination | @subques:={subques}@ans:={ans}@orig_ques:={orig_ques}@title:={title}&===================")
        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/answer_examination.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'answer_examination.txt')).read()
        filled_prompt = prompt_template.format_map({
            "title": title, 
            "question": subques, 
            "answer": ans, 
            "orig_question": orig_ques, 
        })

        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in generating the right response to the question. "},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)

        is_valid = re.search(r'\[Valid Evidence\?\]:\s*(.+)', response).group(1).strip()
        description = re.search(r'\[Description\]:\s*(.+)', response).group(1).strip()
        
        return is_valid, description


    def chain_of_thought(self, question):
        if self.log: 
            print(f"=================== ReasonModule || chain_of_thought | @question:={question}&===================")
        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/cot.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'cot.txt')).read()
        filled_prompt = prompt_template.format_map({
            "Question": question,
        })

        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent assistant designed to handle multi-hop questions. Your primary role is to follow the given instruction to analyze the question and plan the steps to answer it. "},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)
        return response
        

    
    def retrieval_extraction(self, retrieve_info, text, data): 
        if self.log: 
            print(f"=================== ReasonModule || retrieval_extraction | @question:={retrieve_info}@text:={text}@data:={data}&===================")
        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/retrieval_extraction.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'retrieval_extraction.txt')).read()
        filled_prompt = prompt_template.format_map({
            "text": text,
            "description": retrieve_info, 
            "modality": data['modality'], 
            "title": data['title'], 
        })

        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in generating the right response to the question. "},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)
        return response
    

    def retrieval_extraction_all(self, retrieve_info, text, orig_ques = None): 
        if self.log: 
            print(f"=================== ReasonModule || retrieval_extraction_all | @question:={retrieve_info}@text:={text}&===================")
        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/retrieval_extraction_all.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'retrieval_extraction_all.txt')).read()
        filled_prompt = prompt_template.format_map({
            "text": text,
            "description": retrieve_info, 
            "question": orig_ques, 
        })

        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. Your goal is to guide humans in generating effective and useful descriptions in response to given instructions. "},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)
        return response
    

    def instruction_transformation(self, instruction, context, node_type):
        if self.log: 
            print(f"=================== ReasonModule || instruction_transformation | @instruction:={instruction}@text:={context}@node_type:={node_type}&===================")
        
        # instruction_transform
        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/instruction_transform.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'instruction_transform.txt')).read()
        filled_prompt = prompt_template.format_map({
            "description": instruction,
            "context": context, 
            "type": node_type
        })

        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in generating the right response. "},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)
        return response
    

    def answer_examination(self, subques, ans, orig_ques, title):
        if self.log: 
            print(f"=================== ReasonModule || answer_examination | @subques:={subques}@ans:={ans}@orig_ques:={orig_ques}@title:={title}&===================")
        # prompt_template = open('/data1/yihenghu/data_discovery/mmkg_vq/few_shot_examples/answer_examination.txt').read()
        prompt_template = open(os.path.join(self.few_shot_path, 'answer_examination.txt')).read()
        filled_prompt = prompt_template.format_map({
            "title": title, 
            "question": subques, 
            "answer": ans, 
            "orig_question": orig_ques, 
        })

        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in generating the right response to the question. "},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)

        is_valid = re.search(r'\[Valid Evidence\?\]:\s*(.+)', response).group(1).strip()
        description = re.search(r'\[Description\]:\s*(.+)', response).group(1).strip()
        
        return is_valid, description
    

    def find_modality(self, text):
        if self.log: 
            print(f"=================== ReasonModule || find_modality | @text:={text}&===================")
        
        prompt_template = open(os.path.join(self.few_shot_path, 'find_modality.txt')).read()
        filled_prompt = prompt_template.format_map({
            "text": text,
        })
        

        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in generating the right response. "},
            {"role": "user", "content": filled_prompt},
        ]
        model_id = 'llama3-8b-instruct'
        response =  self.run_llm(messages, given_model_id = model_id)
        # return response
        lines = response.strip().split('\n')
        last_lines = lines[-2:]
        last_lines = '\n'.join(last_lines)

        return last_lines
    

    def text_fail_retrieval(self, question, evidence):
        if self.log: 
            print(f"=================== ReasonModule || text_fail_retrieval | @question:={question}@evidence:={evidence}&===================")
        prompt_template = open(os.path.join(self.few_shot_path, 'fail_retrieval.txt')).read()
        filled_prompt = prompt_template.format_map({
            "question": question, 
            "evidence": evidence,
        })
      

        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in generating the right response. "},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)
        return response
    
    def answer_extraction(self, question, answer):
        if self.log: 
            print(f"=================== ReasonModule || answer_extraction | @question:={question}@answer:={answer}&===================")
        prompt_template = open(os.path.join(self.few_shot_path, 'answer_extraction.txt')).read()
        filled_prompt = prompt_template.format_map({
            "question": question, 
            "answer": answer,
        })
        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to complete the given task."},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)
        result = self.extract_string(response, "Rephrase")
        if result is None: 
            return response
        return result


    def relax_answer_production(self, question, evidence): 
        if self.log: 
            print(f"=================== ReasonModule || relax_answer_production | @question:={question}@evidence:={evidence}&===================")
        prompt_template = open(os.path.join(self.few_shot_path, 'relax_answer.txt')).read()
        filled_prompt = prompt_template.format_map({
            "question": question, 
            "evidence": evidence,
        })

        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant to answer a multi-hop question. You need to generate the answer based on the given instruction."},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)
        return response


    def final_answer_production(self, question, evidence): 
        if self.log: 
            print(f"=================== ReasonModule || final_answer_production | @question:={question}@evidence:={evidence}&===================")
        prompt_template = open(os.path.join(self.few_shot_path, 'final_answer.txt')).read()
        filled_prompt = prompt_template.format_map({
            "question": question, 
            "evidence": evidence,
        })


        messages = [
            {"role": "system", "content": "You are a helpful, highly intelligent guided assistant to answer a multi-hop question. You need to generate the answer based on the given instruction."},
            {"role": "user", "content": filled_prompt},
        ]
        response =  self.run_llm(messages)
        return response
        

    def final_answer_generation(self, question, evidence, relax = True):
        response=self.final_answer_production(question,evidence)
        print("\n\n---final answer prod: ", response, "\n\n")
        if 'no answer' in response.lower() and relax:
            response = self.relax_answer_production(question, evidence)
        answer = self.extract_after_keyword(response, 'answer:')
        if answer is None:
            return response
        return answer
        
        

