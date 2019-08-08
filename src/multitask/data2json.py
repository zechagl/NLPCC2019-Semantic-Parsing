import codecs
import re
import json
import os
import argparse
from tqdm import tqdm

'''
transform the raw data of MSParS to json file
'''
def data2json(source, target, label2id):
    labels = set()
    items = []

    with codecs.open(source, 'r', encoding='utf8') as f:
        item = {'id': -1, 'question': [], 'logical': [], 'type': [], 'parameters': [], 'text': ''}

        for line in f:
            if '<question id' in line:
                item['text'] += line
                item['id'] = int(re.search(r'id=(\d+)', line).group(1))
                item['question'] = re.search(r'id=\d+>(.+)', line).group(1).strip().split()
            elif '<logical form' in line:
                item['text'] += line
                item['logical'] = re.search(r'id=\d+>(.+)', line).group(1).strip().split()
            elif '<parameters' in line:
                item['text'] += line
                parameters = re.search(r'id=\d+>(.+)', line).group(1).strip().split('|||')

                for parameter in parameters:
                    p = re.search(r'(.+)\((.+)\).*\[(.+),(.+)\]( @Q\d+)?.*', parameter).groups()
                    item['parameters'].append({'text': p[0].strip(), 'type': p[1].strip(), 'range': (int(p[2]), int(p[3])), 'q': p[4].strip() if p[4] is not None else None})
            elif '<question type' in line:
                item['text'] += line
                item['type'] = re.search(r'id=\d+>(.+)', line).group(1).strip()
                labels.add(item['type'])
                items.append(item)
                item = {'id': -1, 'question': [], 'logical': [], 'type': [], 'parameters': [], 'text': ''}

    label2id_dict = dict([(label, index) for index, label in enumerate(labels)])

    with codecs.open(target, 'w', encoding='utf8') as f:
        json.dump(items, f, indent=2)

    if label2id is not None:
        with codecs.open(label2id, 'w', encoding='utf8') as f:
            json.dump(label2id_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='deal with the raw data of MSParS and transform it to json file')
    parser.add_argument('-s', '--source', help='source file of data')
    parser.add_argument('-t', '--target', help='target file to store json')
    parser.add_argument('-l', '--label2id', help='file to store label2id dict', default=None)
    args = parser.parse_args()
    data2json(args.source, args.target, args.label2id)


if __name__ == '__main__':
    main()
