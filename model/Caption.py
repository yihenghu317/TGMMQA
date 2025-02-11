from transformers import AutoModelForCausalLM, AutoTokenizer
import jsonlines
import pickle
import numpy as np
import os 
import torch
import time

def generate_caption(image_info, model,tokenizer ):
    # checkpoint = "echo840/Monkey"
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda:2', trust_remote_code=True).eval()
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    # tokenizer.padding_side = 'left'
    # tokenizer.pad_token_id = tokenizer.eod_id

    # start_time = time.time()

    updated_data = []
    for i, data in enumerate(image_info):
        img_path = data["path"]

        # question = "Was there a mule on the film poster?"
        # query = f'<img>{img_path}</img> {question} Answer: ' #VQA
        query = f'<img>{img_path}</img> Generate the detailed caption in English: ' #detailed caption

        input_ids = tokenizer(query, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids

        pred = model.generate(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=200,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    # use_cache=True,
                    pad_token_id=tokenizer.eod_id,
                    eos_token_id=tokenizer.eod_id,
                    )
        response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        data['caption'] = response 
        updated_data.append(data)

    return updated_data




def construct_generate_caption(image_info):
    checkpoint = "echo840/Monkey"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='cuda:1', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id

    # start_time = time.time()

    updated_data = []
    for i, data in enumerate(image_info):
        img_path = data["path"]

        query = f'<img>{img_path}</img> Generate the detailed caption in English: ' #detailed caption

        input_ids = tokenizer(query, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids

        pred = model.generate(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=200,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    # use_cache=True,
                    pad_token_id=tokenizer.eod_id,
                    eos_token_id=tokenizer.eod_id,
                    )
        response = tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        data['caption'] = response 
        updated_data.append(data)
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return updated_data