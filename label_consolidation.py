from collections import Counter
from utils import read_json, read_jsonl
from glob import glob
import os
import json

data = []
for json_path in glob('data/*.json'):
    data.append(read_json(json_path))

data_without_error = []
for example in data:
    mark = True
    for annotator in example['data']['labels']:
        if 'data_error' in annotator['step_1'] or 'date_error' in annotator['step_2'] or 'data_error' in annotator['step_3']:
            mark = False
    if mark:
        data_without_error.append(example)

write_file = open('acl_data.jsonl', 'w', encoding='utf-8')
total = 0
count = 0
for example in data_without_error:
    example_labels = {}
    label_counter = {}

    final_label = {}

    streamer_knowledge = [x['streamer_knowledge'] for x in example['data']['labels'] if x['streamer_knowledge'] is not None]
    twitch_knowledge = [x['twitch_knowledge'] for x in example['data']['labels'] if x['twitch_knowledge'] is not None]

    file_name = example['data']['id']
    bad_utterance_file = f'original_data/{file_name}/bad_utterance.jsonl'
    single_user_context_file = f'original_data/{file_name}/single_user_context.jsonl'
    multi_user_context_file = f'original_data/{file_name}/multi_user_context.jsonl'

    bad_utterance = read_json(bad_utterance_file)
    single_user_context = read_jsonl(single_user_context_file)
    multi_user_context = read_jsonl(multi_user_context_file)

    bad_utterance_message = bad_utterance['message']
    single_user_context_messages = [instance['message'] for instance in single_user_context if 'message' in instance]
    multi_user_context_messages = [instance['message'] for instance in multi_user_context if 'message' in instance]

    for step, target in [('step_1', 'target_1'), ('step_2', 'target_2'), ('step_3', 'target_3')]:
        example_labels[step] = []
        for annotator in example['data']['labels']:
            if '판단불가' in annotator[step]:
                annotator[step] = ['Incivility']
            example_labels[step].append(annotator[step])
        label_counter[step] = Counter([item for sublist in example_labels[step] for item in sublist])

        assert len(label_counter[step].most_common(1)) == 1

        consolidated_label = None
        consolidated_target = None
        majority_label, majority_count = label_counter[step].most_common(1)[0]
        if majority_count > 1:
            consolidated_label = majority_label
            if majority_label == 'HIB':
                targets = []
                for annotator in example['data']['labels']:
                    if annotator[target] is not None:
                        targets.append(annotator[target][0])
                majority_target, target_count = Counter(targets).most_common(1)[0]

                if target_count > 1:
                    consolidated_target = majority_target

        final_label[step] = consolidated_label
        final_label[target] = consolidated_target

    final_instance = {
        "id": file_name,
        "message": bad_utterance_message,
        "single_user": single_user_context_messages,
        "multi_user": multi_user_context_messages,
        "label": final_label,
        "streamer_knowledge": streamer_knowledge,
        "twitch_knowledge": twitch_knowledge,
    }

    total += len(multi_user_context_messages)
    count += 1
    write_file.write(json.dumps(final_instance, ensure_ascii=False) + '\n')

print(total/count)