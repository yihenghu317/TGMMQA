
from transformers import AutoModelForCausalLM, AutoTokenizer

# https://github.com/clear-nus/edc/tree/main
oie_prompt_template_file_path = 'prompt_templates/oie_template.txt'
oie_few_shot_example_file_path = './oie_few_shot_examples.txt'


def generate_completion_transformers(
    input: list,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device,
    batch_size=1,
    max_new_token=256,
    answer_prepend="",
):
    tokenizer.pad_token = tokenizer.eos_token
    completions = []
    if isinstance(input, str):
        input = [input]
    for i in range(0, len(input), batch_size):
        batch = input[i : i + batch_size]
        model_inputs = [
            tokenizer.apply_chat_template(entry, add_generation_prompt=True, tokenize=False) + answer_prepend
            for entry in batch
        ]
        model_inputs = tokenizer(model_inputs, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        generated_ids = model.generate(
            **model_inputs, max_new_tokens=max_new_token, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )[:, model_inputs["input_ids"].shape[1] :]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        completions += generated_texts
    return completions

