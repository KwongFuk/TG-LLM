from utlis import *



def my_generate_prompt_CoT_bs(TG, EK, Q):
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



def my_generate_prompt_TG_Reasoning(dataset_name, split_name, TG, EK, Q, CoT, A, f_ICL, Q_type=None, mode=None, eos_token=""):
    '''
    Generate the prompt for the model.

    args:
        dataset_name: string, dataset name
        split_name: string, split name
        TG: list of strings or string, temporal graph
        EK: list of strings or string, exteral knowledge
        Q: string, question
        CoT: list of strings, chain of thought
        A: string, answer
        Q_type: string, question type
        mode: string, mode
        eos_token: string, eos token

    return:
        prompt: string, the prompt
    '''
    if isinstance(TG, list):
        TG = '\n'.join(TG)

    if f_ICL and mode == 'test':
        if dataset_name == 'TGQA':
            Q_type = f'Q{Q_type}'

        if Q_type is None:
            file_path = f'../materials/{dataset_name}/prompt_examples_TGR{split_name}.txt'
        else:
            file_path = f'../materials/{dataset_name}/prompt_examples_TGR_{Q_type}.txt'
        with open(file_path) as txt_file:
            prompt_examples = txt_file.read()

    if f_ICL and mode == 'test':
        prompt = f"Example:\n\n{prompt_examples}\n\nTest:\n\nTimeline:\n{TG}\n\nQuestion: {Q}"
    else:
        prompt = f"Timeline:\n{TG}\n\nQuestion: {Q}"

    if EK is not None:
        if isinstance(EK, list):
            EK = '\n'.join(EK)
        prompt += f"\n\nUseful information:\n{EK}"

    prompt += "\n\nAnswer: Let's think step by step.\n\n"

    if CoT is not None:
        if isinstance(CoT, list):
            CoT = CoT[0]
        prompt += CoT

    prompt += eos_token
    return prompt



def my_generate_prompt_ICL(dataset_name, split_name, learning_setting, story, Q, C, f_ICL, f_shorten_story, f_using_CoT, Q_type=None):
    '''
    Gnerate the prompt for the model

    args:
    story: the story, str
    Q: the question, str
    C: the candidates, list
    Q_type: the question type, str

    return:
    prompt: the generated prompt, str
    '''
    if f_ICL: # use in-context learning
        if dataset_name == 'TGQA':
            Q_type = f'Q{Q_type}'

        if Q_type is None:
            file_path = f'../materials/{dataset_name}/prompt_examples_ICL_{learning_setting}{split_name}.txt'
        else:
            file_path = f'../materials/{dataset_name}/prompt_examples_ICL_{learning_setting}{split_name}_{Q_type}.txt'

        with open(file_path) as txt_file:
            prompt_examples = txt_file.read()

    if f_shorten_story: # shorten the story
        story = shorten_story(story)

    C = add_brackets(C)
    Q += ' Choose from ' + ', '.join(C) + '.'
    
    prompt = f"Example:\n\n{prompt_examples}\n\n\n\nTest:\n\nStory: {story}\n\nQuestion: {Q}" if f_ICL else f"Story: {story}\n\nQuestion: {Q}"
    prompt += "\n\nAnswer: Let's think step by step.\n\n" if f_using_CoT else "\n\nAnswer: "

    return prompt



def my_generate_prompt_TG_trans(dataset_name, story, TG, entities, relation, times, f_ICL, f_shorten_story, f_hard_mode, 
                                transferred_dataset_name, mode=None, eos_token="</s>"):
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