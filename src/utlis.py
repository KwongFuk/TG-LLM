


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