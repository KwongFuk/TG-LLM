import sys
import json
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from utlis import *

os.environ["WANDB_DISABLED"] = "true"







######### Config #########

dataset_selection = 0   # 0: TGQA, 1: TimeQA, 2: TempReason
f_train = 1  # whether train the model
f_test = 1  # whether test the model
f_ICL = 1   # whether use in-context learning during test
f_rewrite = 1  # whether rewrite existing test results
f_shorten_story = 1  # whether shorten the story
f_hard_mode = 0   # whether use hard mode for translation (only know relations) v.s. easy mode (know entities, relations and times)
f_print_example_prompt = True  # whether to print the example prompt for the model
f_unit_test = False  # whether to run the unit test (only for debugging)

# If we want to test the transfer learning performance, just change the transferred dataset name.
# Note: current dataset_name should be 'TGQA', transferred_dataset_name = None (no transfer learning)
transferred_dataset_name = [None, 'TimeQA', 'TempReason'][0]  

###########################



dataset_name = ['TGQA', 'TimeQA', 'TempReason'][dataset_selection]
dataset = load_dataset("sxiong/TGQA", f'{dataset_name}_Story_TG_Trans')


data_train = dataset['train']
data_val = dataset['val']
data_test = dataset['test']


if f_unit_test:
    data_train = create_subset(data_train, 10)
    data_val = create_subset(data_val, 10)
    data_test = create_subset(data_test, 10)


print(data_train)
print(data_val)
print(data_test)






def my_generate_prompt(story, TG, entities, relation, times, mode=None, eos_token="</s>"):
    '''
    Generate the prompt for text to TG translation (given context and keywords, generate the relevant TG)

    Args:
    - story: str or list, the story
    - TG: str or list, the TG
    - entities: str or list, the entities
    - relation: str, the relation
    - times: str or list, the times
    - mode: train or test
    - eos_token: str, the end of sentence token

    Returns:
    - prompt: str, the prompt
    '''

    def add_examples_in_prompt(prompt):
        if f_ICL and mode == 'test':
            file_path = f'../materials/{dataset_name}/prompt_examples_text_to_TG_Trans.txt' if (not f_hard_mode) else \
                        f'../materials/{dataset_name}/prompt_examples_text_to_TG_Trans_hard.txt'
            with open(file_path) as txt_file:
                prompt_examples = txt_file.read()
            prompt = f"\n\n{prompt_examples}\n\nTest:\n{prompt}"
        return prompt.strip()


    # Convert the list to string
    entities = ' , '.join(add_brackets(entities)) if entities is not None else None
    times = ' , '.join(add_brackets(times)) if times is not None else None

    if f_shorten_story:
        story = shorten_story(story)

    if relation is None:
        # If we do not have such information extracted from the questions, we will translate the whole story.
        prompt = add_examples_in_prompt(f"{story}\n\nSummary all the events as a timeline.\n\nTimeline:")
    else:
        if f_hard_mode or entities is None or times is None:
            prompt = add_examples_in_prompt(f"{story}\n\nSummary {relation} as a timeline.\n\nTimeline:")
        else:
            prompt = add_examples_in_prompt(f"{story}\n\nGiven the time periods: {times}, summary {relation} as a timeline. Choose from {entities}.\n\nTimeline:")

    # For training data, we provide the TG as label.
    if TG is not None:
        # Convert the list to string
        TG = '\n'.join(TG)

        # If we want to test the transfer learning performance, we can change the format of the TG in TGQA to other datasets.
        TG = TG_formating_change(TG, dataset_name, transferred_dataset_name)

        prompt += f"\n{TG}\n"

    prompt += eos_token
    return prompt



if f_print_example_prompt:
    for i in range(5):
        if f_train:
            sample = data_train[i]
            prompt = my_generate_prompt(sample['story'], sample['TG'], sample['entities'], sample['relation'], sample['times'], mode='train', eos_token="</s>")
        if f_test:
            sample = data_test[i]
            prompt = my_generate_prompt(sample['story'], None, sample['entities'], sample['relation'], sample['times'], mode='test', eos_token="")
        print(prompt)
        print('===============================')






model_name = "meta-llama/Llama-2-13b-hf"  # can be changed to other models
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )







if f_train:
    # lora config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # this should be set for finutning and batched inference
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))

    # Loading in 8 bit ..."
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    output_dir = f"../model_weights/{dataset_name}_story_TG_trans"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    per_device_eval_batch_size = 4
    eval_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 5e-4
    max_grad_norm = 0.3
    max_steps = 5 if f_unit_test else 50
    warmup_ratio = 0.03
    evaluation_strategy="steps"
    lr_scheduler_type = "constant"

    training_args = transformers.TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                optim=optim,
                evaluation_strategy=evaluation_strategy,
                save_steps=save_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                max_grad_norm=max_grad_norm,
                max_steps=max_steps,
                warmup_ratio=warmup_ratio,
                group_by_length=True,
                lr_scheduler_type=lr_scheduler_type,
                ddp_find_unused_parameters=False,
                eval_accumulation_steps=eval_accumulation_steps,
                per_device_eval_batch_size=per_device_eval_batch_size
            )


    def formatting_func(sample):
        '''
        Given a sample, generate the prompt for the model
        '''
        output = []
        for s, g, e, r, t in zip(sample['story'], sample['TG'], sample['entities'], sample['relation'], sample['times']):
            op = my_generate_prompt(s, g, e, r, t, mode='train')
            output.append(op)

        return output

    # SFT with lora
    trainer = SFTTrainer(
        model=model,
        train_dataset=data_train,
        eval_dataset=data_val,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_args
    )

    # We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()
    trainer.save_model(f"{output_dir}/final")



if f_test:
    def one_batch(model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=512):
        '''
        Given the promot, generate the output and save the results

        Args:
        - model: the model
        - tokenizer: the tokenizer
        - input_prompts: list, the list of prompts
        - samples: list, the list of samples
        - file_paths: list, the list of file paths
        - max_new_tokens: int, the maximum number of tokens to generate

        Returns:
        - None
        '''
        input_tokens = tokenizer(input_prompts, padding='longest', return_tensors="pt")["input_ids"].to("cuda")

        with torch.cuda.amp.autocast():
            generation_output = model.generate(
                input_ids=input_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=0.3,
                repetition_penalty=1.15,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
              )


        for j in range(len(input_prompts)):
            op = tokenizer.decode(generation_output[j], skip_special_tokens=True)
            op = op[len(input_prompts[j]):]
            cur_sample = samples[j]
            cur_sample.update({'prediction': op})

            with open(file_paths[j], 'w') as json_file:
                json.dump(cur_sample, json_file)

        return

    
    # we need padding on the left side to create the embeddings for a whole batch
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    peft_model_id = f"../model_weights/{dataset_name}_story_TG_trans/final"
    peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")
    peft_model.eval()  # Set the model to evaluation mode

    folder_path = f'../results/{dataset_name}_story_TG_trans'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    batch_size = 4
    max_new_tokens = 1024 if dataset_name in 'TGQA' else 512  # Depends on the size of the (relevant) temporal graph

    input_prompts = []
    file_paths = []
    samples = []
    for i in tqdm(range(len(data_test))):
        file_path = folder_path + f'/{str(i)}.json'
        if (os.path.exists(file_path)) and (not f_rewrite):
            continue

        sample = data_test[i]
        cur_prompt = my_generate_prompt(sample['story'], None, sample['entities'], sample['relation'], sample['times'], mode='test', eos_token='')

        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        # collect the prompts as a batch
        if len(input_prompts) >= batch_size:
            one_batch(peft_model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=max_new_tokens)
            input_prompts = []
            file_paths = []
            samples_info = []

    # Last batch that is less than batch_size
    if len(input_prompts) > 0:
        one_batch(peft_model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=max_new_tokens)