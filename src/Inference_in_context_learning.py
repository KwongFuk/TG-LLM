import sys
import os


from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utlis import *
from tqdm import tqdm
from Models import *
from prompt_generation import *

os.environ["WANDB_DISABLED"] = "true"





######### Config #########

dataset_selection = 0  # 0: TGQA, 1: TimeQA_easy, 2: TimeQA_hard, 3: TempReason_l2, 4: TempReason_l3
model_selection = 3  # 0: gpt-3.5-turbo, 1: gpt-4-1106-preview, 2: Llama-2-7b-hf, 3: Llama-2-13b-hf, 4: Llama-2-70b-hf
f_using_CoT = True  # whether use CoT
f_ICL = True   # whether use in-context learning during test
f_rewrite = True  # whether rewrite existing test results
f_shorten_story = True  # whether shorten the story
f_print_example_prompt = True  # whether to print the example prompt for the model
f_unit_test = False  # whether to run the unit test (only for debugging)

###########################


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]
split_name = ['', '_easy', '_hard', '_l2', '_l3'][dataset_selection]
prefix = ['', 'easy_', 'hard_', 'l2_', 'l3_'][dataset_selection]
model_name = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf'][model_selection]
learning_setting = 'SP' if not f_using_CoT else 'CoT'


dataset = load_dataset("sxiong/TGQA", f'{dataset_name}_TGR')
data_test = dataset[prefix + 'test']

if f_unit_test:
    data_test = create_subset(data_test, 10)

print(data_test)



if f_print_example_prompt:
    for i in range(5):
        sample = data_test[i]
        prompt = my_generate_prompt_ICL(dataset_name, split_name, learning_setting, sample['story'], sample['question'], sample['candidates'], 
                                        f_ICL, f_shorten_story, f_using_CoT, Q_type=sample['Q-Type'])

        print(prompt)
        print('===============================')



model = None
tokenizer = None

if 'Llama' in model_name:
    model_name_cmp = f'meta-llama/{model_name}'
    tokenizer = AutoTokenizer.from_pretrained(model_name_cmp)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name_cmp,
                                                load_in_8bit=True,
                                                device_map="auto"
                                                )
    model.eval()



folder_path = f'../results/{dataset_name}_ICL_{learning_setting}{split_name}_{model_name}' if f_ICL else f'../results/{dataset_name}{split_name}_{model_name}'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

batch_size = 4
input_prompts = []
file_paths = []
samples = []
for i in tqdm(range(len(data_test))):
    file_path = f'{folder_path}/{str(i)}.json'
    if os.path.exists(file_path) and (not f_rewrite):
        continue

    sample = data_test[i]
    cur_prompt = my_generate_prompt_ICL(dataset_name, split_name, learning_setting, sample['story'], sample['question'], sample['candidates'], 
                                        f_ICL, f_shorten_story, f_using_CoT, Q_type=sample['Q-Type'])

    input_prompts.append(cur_prompt)
    samples.append(sample)
    file_paths.append(file_path)

    # collect the prompts as a batch
    if len(input_prompts) >= batch_size:
        run_one_batch_ICL(model_name, model, tokenizer, input_prompts, samples, file_paths)
        input_prompts = []
        file_paths = []
        samples = []


if len(input_prompts) > 0:
    run_one_batch_ICL(model_name, model, tokenizer, input_prompts, samples, file_paths)