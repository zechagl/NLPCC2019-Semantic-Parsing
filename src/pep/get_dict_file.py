import codecs
import numpy as np
import json
import argparse


'''
get the dict file, we can get the possibility of 'predicate entity' through dict[entity][predicate]
'''
def get_dict_file(predict_file, tsv_file, dict_file):
    d = dict()

    with codecs.open(predict_file, 'r', encoding='utf8') as f:
        probabilities, _, _ = json.load(f)
        probabilities = np.array(probabilities)[:, 1]

    with codecs.open(tsv_file, 'r', encoding='utf8') as f:
        index = 0

        for line in f:
            line_list = line.strip().split('\t')
            assert len(line_list) == 3
            predicate = line_list[1].strip().split('|')

            for i in range(len(predicate)):
                predicate[i] = '_'.join(predicate[i].strip().split())

            predicate = predicate[0] + ':' + '.'.join(predicate[1:])

            if line_list[2] not in d:
                d[line_list[2]] = dict()

            d[line_list[2]][predicate] = probabilities[index]
            index += 1

    with codecs.open(dict_file, 'w', encoding='utf8') as f:
        json.dump(d, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="get the dict file, we can get the possibility of 'predicate entity' through dict[entity][predicate]")
    parser.add_argument('-p', '--predict', help='predicted possibilities file')
    parser.add_argument('-t', '--tsv', help='tsv file used for prediction')
    parser.add_argument('-d', '--dict', help='dict file generated')
    args = parser.parse_args()
    get_dict_file(args.predict, args.tsv, args.dict)


if __name__ == '__main__':
    main()
