import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from utlis import *
from Models import *
from prompt_generation import *
import argparse

os.environ["WANDB_DISABLED"] = "true"




parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str)
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--rewrite', action='store_true')
parser.add_argument('--ICL', action='store_true')
parser.add_argument('--shorten_story', action='store_true')
parser.add_argument('--hard_mode', action='store_true')
parser.add_argument('--print_prompt', action='store_true')
parser.add_argument('--unit_test', action='store_true')
parser.add_argument('--transferred_dataset', type=str)
parser.add_argument('--transferred', action='store_true')

args = parser.parse_args()



######### Config #########

dataset_name = args.dataset   # 'TGQA', 'TimeQA', 'TempReason'
f_train = args.train   # whether train the model
f_test = args.test  # whether test the model
f_rewrite = args.rewrite  # whether rewrite existing test results
f_ICL = args.ICL  # whether use in-context learning during test
f_shorten_story = args.shorten_story   # whether shorten the story (For TimeQA and TempReason, it is possible that the story is too long to feed into the model)
f_hard_mode = args.hard_mode   # whether use hard mode (only know relations) v.s. easy mode (know entities, relations and times) for translation
f_print_example_prompt = args.print_prompt  # whether to print the example prompt for the model
f_unit_test = args.unit_test   # whether to run the unit test (only for debugging)

# If we want to test the transfer learning performance, just change the transferred dataset name.
# Note: current dataset_name should be 'TGQA', transferred_dataset_name = None (no transfer learning)
transferred_dataset_name = args.transferred_dataset
f_transferred = args.transferred  # whether to use transfer learning during test (if True, we will read the model weights learned from the transferred dataset)

###########################


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



if f_print_example_prompt:
    for i in range(5):
        if f_train:
            sample = data_train[i]
            prompt = my_generate_prompt_TG_trans(dataset_name, sample['story'], sample['TG'], sample['entities'], sample['relation'], sample['times'], 
                                                 f_ICL, f_shorten_story, f_hard_mode, transferred_dataset_name, mode='train', eos_token="</s>")
        if f_test:
            sample = data_test[i]
            prompt = my_generate_prompt_TG_trans(dataset_name, sample['story'], None, sample['entities'], sample['relation'], sample['times'], 
                                                 f_ICL, f_shorten_story, f_hard_mode, transferred_dataset_name, mode='test', eos_token="")
        print(prompt)
        print('===============================')



model_name = "meta-llama/Llama-2-13b-hf"  # can be changed to other models
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )



if f_train:
    def formatting_func(sample):
        '''
        Given a sample, generate the prompt for the model
        '''
        output = []
        for s, g, e, r, t in zip(sample['story'], sample['TG'], sample['entities'], sample['relation'], sample['times']):
            op = my_generate_prompt_TG_trans(dataset_name, s, g, e, r, t, f_ICL, f_shorten_story, f_hard_mode, 
                                            transferred_dataset_name, mode='train')
            output.append(op)

        return output

    output_dir = f"../model_weights/{dataset_name}_story_TG_trans"
    if transferred_dataset_name is not None:
        output_dir = f"../model_weights/{dataset_name}_to_{transferred_dataset_name}_story_TG_trans"
    SFT_with_LoRA(model, tokenizer, output_dir, f_unit_test, formatting_func, data_train, data_val, 4, 4096)


if f_test:
    # we need padding on the left side to create the embeddings for a whole batch
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    peft_model_id = f"../model_weights/{dataset_name}_story_TG_trans/final"
    if f_transferred:
        peft_model_id = f"../model_weights/TGQA_to_{dataset_name}_story_TG_trans/final"
    peft_model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")
    peft_model.eval()  # Set the model to evaluation mode

    folder_path = f'../results/{dataset_name}_story_TG_trans'
    if f_transferred:
        folder_path = f'../results/TGQA_to_{dataset_name}_story_TG_trans'
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
        cur_prompt = my_generate_prompt_TG_trans(dataset_name, sample['story'], None, sample['entities'], sample['relation'], sample['times'], 
                                                 f_ICL, f_shorten_story, f_hard_mode, transferred_dataset_name, mode='test', eos_token='')

        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

 
        # collect the prompts as a batch
        if len(input_prompts) >= batch_size:
            run_one_batch_generation(peft_model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=max_new_tokens)
            input_prompts = []
            file_paths = []
            samples = []

    # Last batch that is less than batch_size
    if len(input_prompts) > 0:
        run_one_batch_generation(peft_model, tokenizer, input_prompts, samples, file_paths, max_new_tokens=max_new_tokens)