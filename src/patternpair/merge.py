import codecs
from tqdm import tqdm
import json
import os
import argparse
import numpy as np
import sys


def merge_json(path, save_path):
    assert os.path.isdir(path) == True
    files = os.listdir(path)
    files.sort()
    probabilities = []
    predict = []
    gold = []

    for file in tqdm(files):
        with codecs.open(os.path.join(path, file), 'r', encoding='utf8') as f:
            c1, c2, c3 = json.load(f)

        probabilities.extend(c1)
        predict.extend(c2)
        gold.extend(c3)

    with codecs.open(save_path, 'w', encoding='utf8') as f:
        json.dump((probabilities, predict, gold), f, indent=2)


def merge(mode):
    files = ['7epoch_base/7_base_%s.json', '7epoch_udf/7_udf_%s.json', 'ep6/ep6_%s.json', 'ep9/ep9_%s.json', 'epc/epc_%s.json']
    files = list(map(lambda x: '../../output/patternpair/' + x % mode, files))
    probabilities = []
    predict = []
    gold = []

    for index, file in enumerate(tqdm(files)):
        print(file)
        with codecs.open(file, 'r', encoding='utf8') as f:
            c1, c2, c3 = json.load(f)
        
        if index == 0:
            probabilities = np.zeros_like(c1)
            gold = c3
        
        c1 = np.array(c1)
        probabilities += c1
        print(gold == c3)

    probabilities /= len(files)
    probabilities = probabilities.tolist()
    print(len(files))

    with codecs.open('../../data/json/pattern_pair_merge_5_dev.json', 'w', encoding='utf8') as f:
        json.dump((probabilities, predict, gold), f, indent=2)


def main():
    #argparser = argparse.ArgumentParser('merge json')
    #argparser.add_argument('-p', '--path', help='path of json files to be merged')
    #argparser.add_argument('-f', '--file', help='merged json file')
    #args = argparser.parse_args()
    #merge_json(args.path, args.file)
    merge('dev')


if __name__ == '__main__':
    main()
