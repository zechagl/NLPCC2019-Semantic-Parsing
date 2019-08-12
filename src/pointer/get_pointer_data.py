import json
import codecs
import re
import itertools
import numpy as np
from tqdm import tqdm
import argparse
import os


def get_pointer_data(input_file, output_file):
    with codecs.open('merge.merged_7_test_todo3.json', 'r', encoding='utf8') as f:
        data = json.load(f)

    new_data = []

    for index, item in enumerate(tqdm(data)):
        question = ' '.join(item['question'])
        logical = []

        for pred in item['logical_pred']:
            logical.append(pred[0])

        if 'logical_pred_0' in item and 'logical_pred_1' in item:
            for pred_0, pred_1 in itertools.product(item['logical_pred_0'], item['logical_pred_1']):
                logical.append(pred_0[0] + ' ||| ' + pred_1[0])

        new_logical = []

        for l in logical:
            l_list = l.split()
            new_l = []

            for m in l_list:
                new_l += re.split(r'([.:_])', m)

            new_logical.append(' '.join(new_l))

        new_data.append([question, new_logical])

    with codecs.open('merge_7_test.json', 'w', encoding='utf8') as f:
        json.dump(new_data, f, indent=2)
