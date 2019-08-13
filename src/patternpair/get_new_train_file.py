import codecs
import json
from tqdm import tqdm
import numpy as np
import random
import argparse


'''
write new train_file
'''
def get_data(select_file, new_train, train_full):
    with codecs.open(select_file, 'r', encoding='utf8') as f:
        data = json.load(f)

    with codecs.open(train_full, 'r', encoding='utf8') as f:
        lines = f.readlines()

    with codecs.open(new_train, 'w', encoding='utf8') as f:
        for i in data:
            for j in i:
                f.write(lines[j].strip() + '\n')


'''
there may be:
1   pattern0    pattern1
0   pattern0    pattern2
0   pattern0    pattern3
...
1   pattern4    pattern5
0   pattern4    pattern6
...
1   pattern0    pattern3
'pattern0 pattern3' may be positive in one case while negative in another case
so pattern3 shouldn't be selected as negative samples for pattern0 when we encounter '1 pattern0 pattern1'
we find line numbers labeled as '0' while its pattern pair is labeled '1' in another case
we won't select negative samples from these line numbers
'''
def generate_line_numbers(train_file, line_number_file):
    with codecs.open(train_file, 'r', encoding='utf8') as f:
        lines = f.readlines()

    gold_set = set()
    gold_line_number = []

    for line in tqdm(lines, 'reading gold set'):
        line_list = line.strip().split('\t')

        if line_list[0] == '1':
            gold_set.add(line_list[1] + '+' + line_list[2])

    for index, line in enumerate(tqdm(lines, 'reading gold line number')):
        line_list = line.strip().split('\t')

        if (line_list[1] + '+' + line_list[2]) in gold_set and line_list[0] == '0':
            gold_line_number.append(index)

    with codecs.open(line_number_file, 'w', encoding='utf8') as f:
        json.dump(gold_line_number, f, indent=2)


'''
select suitable negative samples
process in 3 parallel process to accelerate
'''
def get_new_train_file(line_number_file, train_predict, mode, select_file):
    value = 0.0001

    with codecs.open(line_number_file, 'r', encoding='utf8') as f:
        gold_line_numbers = json.load(f)

    with codecs.open(train_predict, 'r', encoding='utf8') as f:
        possibilities, predict, gold = json.load(f)

    possibilities = np.array(possibilities)[:, 1]
    predict = np.array(predict)
    gold = np.array(gold)
    assert gold[0] == 1
    assert gold[-1] == 0
    index_list = np.where(gold == 1)[0] # labeled as '1'
    length_list = []
    n = 0
    n1 = 0
    n2 = 0

    if mode == 0:
        s = 0
        e = 27366
    elif mode == 1:
        s = 27366
        e = 54732
    else:
        s = 54732
        e = len(index_list)

    # too slow! we split it into 3 parts so that we can parallel 3 process to accelerate
    for index, start in enumerate(tqdm(index_list[s:e])):
        if index == len(index_list) - 1:
            end = len(gold)
        else:
            end = index_list[index + 1]

        cnt = 0
        candidates_plus = []
        candidates_minus = []

        if possibilities[start] >= value:
            n += 1

        for i in range(start + 1, end):
            if possibilities[i] >= value and i not in gold_line_numbers:
                cnt += 1
                candidates_plus.append(int(i))
            elif possibilities[i] < value and i not in gold_line_numbers:
                candidates_minus.append(int(i))

        num_plus = min(20, len(candidates_plus))
        num_minus = min(5, len(candidates_minus))
        select_plus = random.sample(range(len(candidates_plus)), num_plus)
        select_minus = random.sample(range(len(candidates_minus)), num_minus)
        selected = []

        for i in select_plus:
            selected.append(candidates_plus[i])

        for i in select_minus:
            selected.append(candidates_minus[i])

        random.shuffle(selected)
        selected.append(int(start))
        random.shuffle(selected)
        length_list.append(selected)
        n1 += cnt
        n2 += end - start - 1

    random.shuffle(length_list)
    print(len(length_list))
    print(n / len(index_list))
    print(n1 / n2)
    print(n1)
    print(n2)

    with codecs.open(select_file + str(mode), 'w', encoding='utf8') as f:
        json.dump(length_list, f, indent=2)


'''
merge three select files
'''
def merge(select_file):
    with codecs.open(select_file + '0', 'r', encoding='utf8') as f:
        data0 = json.load(f)

    with codecs.open(select_file + '1', 'r', encoding='utf8') as f:
        data1 = json.load(f)

    with codecs.open(select_file + '2', 'r', encoding='utf8') as f:
        data2 = json.load(f)

    merged = data0 + data1 + data2
    random.shuffle(merged)

    with codecs.open(select_file, 'w', encoding='utf8') as f:
        json.dump(merged, f)


def main():
    argparser = argparse.ArgumentParser('get new train file using ranking sampling before every training epoch for pattern pair matching net')
    argparser.add_argument('--train_full', help='full train file path')
    argparser.add_argument('--lines', help='line numbers file path')
    argparser.add_argument('--train_predict', help='predict the full train file and get its possibilities')
    argparser.add_argument('--mode', help='l for generate_line_numbers, n for get_new_train_file, 012 for 3 parallel process, t for get new train file')
    argparser.add_argument('--select', help='select file path')
    argparser.add_argument('--train_new', help='new train file path')
    args = argparser.parse_args()

    if 'l' in args.mode:
        generate_line_numbers(args.train_full, args.lines)

    if 'n' in args.mode:
        if '0' in args.mode:
            get_new_train_file(args.lines, args.train_predict, 0, args.select)
        elif '1' in args.mode:
            get_new_train_file(args.lines, args.train_predict, 1, args.select)
        else:
            get_new_train_file(args.lines, args.train_predict, 2, args.select)

    if 't' in args.mode:
        merge(args.select)
        get_data(args.select, args.train_new, args.train_full)


if __name__ == '__main__':
    main()
