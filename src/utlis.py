import random
import os
import copy
from datasets import load_dataset
import sys



def add_brackets(ls):
    '''
    Add brackets to the elements in the list
    
    Args:
    - ls: list, the list of elements

    Returns:
    - ls: list, the list of elements with brackets
    '''
    if '(' != ls[0].strip()[0] or ')' != ls[0].strip()[-1]:
        ls = [f'( {e.strip()} )' for e in ls]
    return ls


def TG_formating_change(TG, dataset_name, targ_dataset):
    '''
    To test transfer learning performance, change the format of the TG from TGQA to other datasets.

    Args:
    - TG: str, the TG
    - dataset_name: str, the ori dataset
    - targ_dataset: str, the target dataset

    Returns:
    - TG: str, the TG with the new format
    '''
    assert (targ_dataset is None) or (targ_dataset in ['TimeQA', 'TempReason'] and dataset_name == 'TGQA') 
    
    if targ_dataset is None:
        return TG

    event_pos = {}
    timeline = []
    eventline = []
    cnt = 0
    for line in TG.split('\n'):
        if not len(line.strip()):
            continue
        if 'starts at' in line:
            event = line.split('starts at')[0].strip()
            time = line.split('starts at')[1].strip()
            event_pos[event] = cnt
            eventline.append(event)
            timeline.append(time)
            cnt += 1
        else:
            event = line.split('ends at')[0].strip()
            time = line.split('ends at')[1].strip()
            if event in event_pos:
                timeline[event_pos[event]] = timeline[event_pos[event]] + ' - ' + time
            else:
                event_pos[event] = cnt
                eventline.append(event)
                timeline.append(time)
                cnt += 1

    TG = []
    for event, time in zip(eventline, timeline):
        if '-' not in time:
            time = time + ' - ' + time
        if targ_dataset == 'TempReason':
            time = 'from ' + time.replace(' - ', ' to ')

        if targ_dataset == 'TimeQA':
            TG.append(f'{time} : {event}')
        else:
            TG.append(f'{event[1:-1]} {time}.')
    TG = '\n'.join(TG)
    
    return TG


def process_CoT(ans_pred):
    '''
    Remove the final answer from the CoT since we need to calculate the perplexity of the candidates.
    '''
    ans_pred = ans_pred.strip()
    for identifier in [' the answer is ', 'Answer:', ' answer is:', ' the correct answer is', ' the answers are ']:
        if identifier in ans_pred:
            ans_pred = ans_pred.split(identifier)[0].strip()
            break
    return ans_pred + ' the answer is '


def process_id(dataset_name, sample_id):
    '''
    Process the sample id to obtain its story id.
    '''
    if dataset_name == 'TGQA':
        story_id = sample_id.split('_')[0]
    if dataset_name == 'TimeQA':
        story_id = sample_id.split('_easy_')[0] if '_easy_' in sample_id else sample_id.split('_hard_')[0]
    if dataset_name == 'TempReason':
        story_id = sample_id[3:-2]
    return story_id


def data_augmentation(dataset_name, TG, EK, Q, CoT, C, A, flag_rm_irr_edges=False, flag_change_relations=False, flag_change_entities=False, flag_change_times=False, extra_data=None):
    '''
    Augment the data by randomly deleting irrelevant edges and changing the entities, relations, and times.

    args:
        dataset_name: string, dataset name
        TG: list of strings or string, temporal graph
        EK: list of strings or string, exteral knowledge
        Q: string, question
        CoT: list of strings, chain of thought
        C: list of strings, candidates
        A: string, answer
        flag_rm_irr_edges: bool, whether to remove irrelevant edges
        flag_change_relations: bool, whether to change relations
        flag_change_entities: bool, whether to change entities
        flag_change_times: bool, whether to change times
        extra_data: tuple, extra data for data augmentation

    return:
        TG: list of strings, augmented temporal graph
        EK: list of strings, augmented exteral knowledge
        Q: string, augmented question
        CoT: list of strings, augmented chain of thought
        C: list of strings, augmented candidates
        A: string, augmented answer
    '''
    if extra_data is not None:
        rel_entity_dict, global_ent_mapping, global_names_cnt, random_entity_names, global_time_offset = extra_data

    if flag_rm_irr_edges:
        TG = rm_irr_edges(TG, CoT, Q)

    if flag_change_entities:
        TG, EK, Q, CoT, C, A = change_entities(TG, [TG, EK, Q, CoT, C, A], rel_entity_dict, global_ent_mapping, global_names_cnt, random_entity_names)
    
    if flag_change_times:
        TG, EK, Q, CoT, C, A = change_times(TG, [TG, EK, Q, CoT, C, A], global_time_offset)

    if flag_change_relations:
        TG = change_rels(dataset_name, TG)

    return TG, EK, Q, CoT, C, A



def rm_irr_edges(TG, CoT, Q, ratio=0.2):
    '''
    Remove irrelevant edges in the temporal graph.
    '''
    rm_indices = [i for (i, fact) in enumerate(TG) if (fact[:-len(fact.split(')')[-1])].strip() not in CoT) and (fact[:-len(fact.split(')')[-1])].strip() not in Q)]
    random.shuffle(rm_indices)
    rm_indices = rm_indices[:int(ratio*len(rm_indices))]
    facts_ls = [fact for (i, fact) in enumerate(TG) if i not in rm_indices]
    return facts_ls


def change_rels(dataset_name, TG):
    '''
    Change relations in the temporal graph.
    '''
    with open(f'../materials/{dataset_name}/rel_synonyms.txt', 'r') as file:
        contents = file.readlines()

    rel_synonyms = {}
    for line in contents:
        if len(line.strip()) > 0:
            rel = line.split('\t')[0].strip()
            rel_synonyms[rel] = line.split('\t')

    facts_new_ls = []
    for fact in TG:
        fact_front = fact[:-len(fact.split(')')[-1])].strip()[1:-1]
        fact_back = fact[-len(fact.split(')')[-1]):]
        for rel in rel_synonyms:
            if ' ' + rel in fact_front:
                sub = fact_front.split(rel)[0].strip()
                obj = fact_front.split(rel)[1].strip()
                fact_new = f'({sub} {random.choice(rel_synonyms[rel]).strip()} {obj}){fact_back}'
                facts_new_ls.append(fact_new)
                break

    return facts_new_ls



def change_times(TG, elements, global_time_offset):
    '''
    Change times in the temporal graph.
    '''
    def replace_time(e, mapping_time):
        for time in mapping_time:
            e = e.replace(time, mapping_time[time])
        e = e.replace('_', '')
        return e

    mapping_time = {}
    for fact in TG:
        time = fact.split(' at ')[-1]
        mapping_time[time] = str(int(time) + global_time_offset)
        mapping_time[time] = mapping_time[time][0] + '_' + mapping_time[time][1:]

    elements_new = []
    for e in elements:
        if isinstance(e, list):
            e_new = [replace_time(sub_e, mapping_time) for sub_e in e]
        else:
            e_new = replace_time(e, mapping_time)
        elements_new.append(e_new)

    return elements_new



def collect_entity(dataset_name):
    '''
    Collect subject and object entities for each relation.

    return:
        rel_entity_dict: dict, subject and object entities for each relation
    '''
    with open(f'../materials/{dataset_name}/rel_dict.txt', 'r') as file:
        contents = file.readlines()

    rel_entity_dict = {}
    for line in contents:
        if len(line.strip()) > 0:
            rel = line.split('\t')[0].strip()
            rel_entity_dict[rel] = {'sub': [], 'obj': []}

    def process_ent(rel_entity_dict, data):
        for i in range(len(data)):
            sample = data[i]
            for fact in sample['TG']:
                fact = fact[:-len(fact.split(')')[-1])].strip()[1:-1]
                for rel in rel_entity_dict:
                    if ' ' + rel in fact:
                        rel_entity_dict[rel]['sub'].append(fact.split(rel)[0].strip())
                        rel_entity_dict[rel]['obj'].append(fact.split(rel)[1].strip())
                        break
        return rel_entity_dict

    dataset = load_dataset("sxiong/TGQA", f'{dataset_name}_TGR')
    for split in dataset:
        data = dataset[split]
        rel_entity_dict = process_ent(rel_entity_dict, data)

    for rel in rel_entity_dict:
        rel_entity_dict[rel]['sub'] = list(set(rel_entity_dict[rel]['sub']))
        rel_entity_dict[rel]['obj'] = list(set(rel_entity_dict[rel]['obj']))

    sub_total = []
    for rel in rel_entity_dict:
        sub_total += rel_entity_dict[rel]['sub']
    sub_total = list(set(sub_total))
    rel_entity_dict['sub_total'] = sub_total

    return rel_entity_dict



def collect_entity_v2(dataset_name, existing_names=None):
    '''
    Collect random subject and object entity names for each relation.

    args:
        existing_names: dict, existing entity names (which we want to exclude)

    return:
        ent_names: dict, random entity names
    '''
    path = f'../materials/{dataset_name}/random_names'
    with open(f'{path}/sub_total.txt') as file:
        context = file.readlines()

    sub_total = [line.strip() for line in context]
    sub_total = list(set(sub_total))

    ent_names = {}
    for file_path in os.listdir(path):
        if file_path != 'sub_total.txt':
            with open(f'{path}/{file_path}') as file:
                context = file.read()
            rel = file_path.split('.')[0]
            ent_names[rel] = {}
            ent_names[rel]['obj'] = context.strip().split('\n')
            ent_names[rel]['obj'] = list(set(ent_names[rel]['obj']))


    if existing_names is not None:
        sub_total = [name for name in sub_total if name not in existing_names['sub_total']]
        random.shuffle(sub_total)
        for rel in ent_names:
            ent_names[rel]['obj'] = [name for name in ent_names[rel]['obj'] if name not in existing_names[rel]['obj']]
            random.shuffle(ent_names[rel]['obj'])
        ent_names['sub_total'] = sub_total

    return ent_names


def change_entities(TG, elements, rel_entity_dict, global_ent_mapping, global_names_cnt, random_entity_names):
    '''
    Change entities in the temporal graph.
    
    args:
        TG: list of strings, temporal graph
        elements: list of strings or list of list of strings, elements

    return:
        elements_new: list of strings or list of list of strings, augmented elements
    '''
    for fact in TG:
        fact = fact[:-len(fact.split(')')[-1])].strip()[1:-1]
        
        # Find the relation
        for rel in rel_entity_dict:
            if ' ' + rel not in fact:
                continue
            
            # deal with the subject
            sub = fact.split(rel)[0].strip()
            if sub not in global_ent_mapping:
                if 'sub_total' not in global_names_cnt:
                    global_names_cnt['sub_total'] = 0
                global_ent_mapping[sub] = random_entity_names['sub_total'][global_names_cnt['sub_total']]
                global_names_cnt['sub_total'] += 1

            # deal with the object
            obj = fact.split(rel)[1].strip()
            if obj not in global_ent_mapping:
                if rel in ['was married to']:
                    if 'sub_total' not in global_names_cnt:
                        global_names_cnt['sub_total'] = 0
                    global_ent_mapping[obj] = random_entity_names['sub_total'][global_names_cnt['sub_total']]
                    global_names_cnt['sub_total'] += 1
                else:
                    if rel not in global_names_cnt:
                        global_names_cnt[rel] = 0

                    valid_name = random_entity_names[rel]['obj'][global_names_cnt[rel]]
                    while valid_name in global_ent_mapping.values():
                        global_names_cnt[rel] += 1
                        valid_name = random_entity_names[rel]['obj'][global_names_cnt[rel]]

                    global_ent_mapping[obj] = copy.copy(valid_name)

            break
    
 
    def replace_entity(e, mapping_ent):
        for entity in mapping_ent:
            e = e.replace(entity, mapping_ent[entity])
        return e

    elements_new = []
    for e in elements:
        if isinstance(e, list):
            e_new = [replace_entity(sub_e, global_ent_mapping) for sub_e in e]
        else:
            e_new = replace_entity(e, global_ent_mapping)
        elements_new.append(e_new)
    return elements_new


def CoT_sampling(CoT, CoT_sample_prob):
    '''
    Sample a chain of thought according to the given probabilities.
    '''
    return random.choices(CoT, weights=CoT_sample_prob, k=1)


def create_subset(dataset, size, shuffle=False, seed=None):
    '''
    Create a subset of the dataset.
    '''
    if size == -1:
        return dataset
    
    # Define the indices for the subset
    indices = list(range(len(dataset)))[:size]

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    subset = dataset.select(indices)
    return subset


def shorten_story(story):
    '''
    Shorten the story.
    '''
    return ' '.join(story.split(' ')[:1000])  # simply shorten the story to 1000 words


def parse_TG_pred(pred):
    for end_identifier in ['Test:']:
        if end_identifier in pred:
            pred = pred.split(end_identifier)[0].strip()
        break
    return pred


def replace_empty_answer_as_unknown(ans):
    return [a if len(a)>0 else 'Unknown' for a in ans]