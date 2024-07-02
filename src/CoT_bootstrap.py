import sys
import json
import os
import numpy as np


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utlis import *
from tqdm import tqdm
import itertools

os.environ["WANDB_DISABLED"] = "true"









######### Config #########

dataset_selection = 0  # 0: TGQA, 1: TimeQA_easy, 2: TimeQA_hard, 3: TempReason_l2, 4: TempReason_l3
f_print_example_prompt = True  # whether to print the example prompt for the model
f_unit_test = False  # whether to run the unit test (only for debugging)

###########################


dataset_name = ['TGQA', 'TimeQA', 'TimeQA', 'TempReason', 'TempReason'][dataset_selection]

dataset = load_dataset("sxiong/TGQA", f'{dataset_name}_TGR')

split_train = ['train', 'easy_train', 'hard_train', 'l2_train', 'l3_train'][dataset_selection]
data_train = dataset[split_train]

split_val = ['val', 'easy_val', 'hard_val', 'l2_val', 'l3_val'][dataset_selection]
data_val = dataset[split_val]


if f_unit_test:
    data_train = create_subset(data_train, 10)
    data_val = create_subset(data_val, 10)


print(data_train)
print(data_val)






def my_generate_prompt(TG, EK, Q):
    '''
    Generate the prompt for the model
    
    Args:
    TG: list of strings, temporal graph
    EK: list of strings, external knowledge
    Q: string, the question
    
    Returns:
    prompt: string, the prompt for the model
    '''
    TG = '\n'.join(TG)

    prompt = f"Timeline:\n{TG}\n\nQuestion: {Q}"

    if EK is not None:
        EK = '\n'.join(EK)
        prompt += f"\n\nUseful information:\n{EK}"

    prompt += "\n\nAnswer: Let's think step by step.\n\n"

    return prompt



if f_print_example_prompt:
    for i in range(5):
        sample = data_train[i]
        prompt = my_generate_prompt(sample['TG'], sample['external knowledge'], sample['question'])
        print(prompt)
        print('===============================')





model_name = "meta-llama/Llama-2-13b-hf" # you can change to other models
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token_id = 0
tokenizer.padding_side = 'right'


model = AutoModelForCausalLM.from_pretrained(model_name,
                                            load_in_8bit=True,
                                            device_map="auto"
                                            )

model.eval()  # Set the model to evaluation mode






def one_batch(input_prompts, samples):
    '''
    For each sample, calculate the contrastive score for each CoT. Then save the results to the corresponding files.
    
    Args:
    input_prompts: the input prompts, list
    samples: the samples, dict

    Returns:
    samples: the samples with the CoT sample probability, dict
    '''
    gamma = 0.5    # score = logProbs_pos + gamma*(logProbs_pos - logProbs_neg)
    for j in range(len(input_prompts)):
        cur_sample = samples[j]

        # Prepare the combinations of CoT and candidates
        neg_ans = [cand for cand in cur_sample['candidates'] if cand not in cur_sample['answer']]
        combinations = list(itertools.product(cur_sample['CoT'], neg_ans + cur_sample['answer']))
        cur_prompts = []
        context_len = []
        for comb in combinations:
            context = input_prompts[j] + process_CoT(comb[0])
            cur_prompts.append(context + comb[1])
            
            len_bf = tokenizer(context, return_tensors="pt")["input_ids"].shape[1]
            len_af = tokenizer(context + comb[1], return_tensors="pt")["input_ids"].shape[1]
            
            # The length of the context should be at least 1 less than the length of all the tokens
            context_len.append(min(len_bf, len_af-1))
        

        # Tokenize the entire batch of answers at once with truncation
        input_tokens = tokenizer(cur_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)["input_ids"].to("cuda")        

        # Create target_ids with masked context
        target_ids = input_tokens.clone()
        for i in range(len(context_len)):
            target_ids[i, :context_len[i]] = -100  # mask the context before the answer

        # Mask padding tokens
        padding_mask = input_tokens == tokenizer.pad_token_id
        target_ids[padding_mask] = -100  # mask the padding tokens

        # # Verify target_ids
        # print("Target IDs after padding mask:", target_ids)

        # Process the batch
        with torch.no_grad():
            outputs = model(input_tokens, labels=target_ids)
            logits = outputs.logits

        # Calculate loss per token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Reshape loss to match input tokens shape
        loss = loss.view(shift_labels.size())

        # Mask loss for padding tokens
        loss[padding_mask[:, 1:]] = 0.0

        # # Verify loss tensor
        # print("Loss tensor:", loss)

        # Aggregate loss for each answer
        valid_counts = (loss != 0).sum(dim=1)
        valid_counts[valid_counts == 0] = 1  # Avoid division by zero
        loss_per_answer = loss.sum(dim=1) / valid_counts
        loss_per_answer = loss_per_answer.cpu().numpy()

        # Split the losses back to individual CoTs
        loss_per_answer = loss_per_answer.reshape((len(cur_sample['CoT']), -1))
        logProbs_pos = np.mean(loss_per_answer[:, len(neg_ans):], axis=1)
        logProbs_neg = np.mean(loss_per_answer[:, :len(neg_ans)], axis=1)

        # print("Loss per answer:", loss_per_answer)
        # print("Log Probs Pos:", logProbs_pos)
        # print("Log Probs Neg:", logProbs_neg)

        # Constrastive score:
        scores = logProbs_pos + gamma*(logProbs_pos - logProbs_neg)
        scores = [np.exp(-10*s) for s in scores]

        # Normalize the scores to get the probability
        cur_sample['CoT_sample_prob'] = (scores/np.sum(scores)).tolist()

    return samples



def CoT_bootstrap(data, filename):
    '''
    Given a list of CoT for each sample that leads to the correct final answer, calculate the probability of each CoT for each sample.
    Todo: Here we start from the data with filtered CoTs. We can also start from the data with no CoTs, and we need to generate and filter the CoTs in this function.

    Args:
    data: the data with filtered CoTs
    filename: the filename to save the results

    Returns:
    None
    '''
    folder_path = f'../results/{dataset_name}_TGR_CoT_bs'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    batch_size = 4

    data_new = []
    input_prompts = []
    input_samples = []

    for sample in tqdm(data):
        cur_prompt = my_generate_prompt(sample['TG'], sample['external knowledge'], sample['question'])
        input_prompts.append(cur_prompt)
        input_samples.append(sample)

        if len(input_prompts) >= batch_size:
            samples = one_batch(input_prompts, input_samples)
            data_new += samples
            input_prompts = []
            input_samples = []

    # Last batch that is less than batch_size
    if len(input_prompts) > 0:
        samples = one_batch(input_prompts, input_samples)
        data_new += samples


    file_path = f'{folder_path}/{filename}.json'
    with open(file_path, 'w') as json_file:
        json.dump(data_new, json_file)

    return



CoT_bootstrap(data_train, split_train)