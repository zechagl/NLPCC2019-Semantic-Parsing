import codecs
import json
import argparse
import random
from tqdm import tqdm


'''
get the gold entity labeling of a MSParS example
labels: 'b', 'm', 'o'
'''
def get_entity_labeling(item):
    entity_labeling = ['o'] * len(item['question'])

    for parameter in item['parameters']:
        if parameter['type'] == 'entity':
            begin, end = parameter['range']

            if item['type'] == 'multi-turn-predicate' and parameter['q'] == '@Q2':
                assert '|||' in item['question']
                bias = item['question'].index('|||') + 1
                begin += bias
                end += bias

            entity_labeling[end] = 'm'
            entity_labeling[begin] = 'b'

            for i in range(begin + 1, end):
                entity_labeling[i] = 'm'

    return ''.join(entity_labeling)


'''
get multi classes json file, and generate train file and test file(from dev file) for multitask model
'''
def generate_multi_task_data(source_train, source_dev, target_train, target_test, label2id, seed):
    with codecs.open(source_train, 'r', encoding='utf8') as f:
        train_data = json.load(f)

    with codecs.open(source_dev, 'r', encoding='utf8') as f:
        test_data = json.load(f)

    with codecs.open(label2id, 'r', encoding='utf8') as f:
        label2id = json.load(f)

    train_question = []
    train_class = []
    train_entity = []
    test_question = []
    test_class = []
    test_entity = []

    for item in tqdm(train_data, 'train_data'):
        train_question.append(' '.join(item['question']))
        train_class.append(label2id[item['type']])
        train_entity.append(get_entity_labeling(item))

    for item in tqdm(test_data, 'test_data'):
        test_question.append(' '.join(item['question']))
        test_class.append(label2id[item['type']])
        test_entity.append(get_entity_labeling(item))

    # shuffle training data because bert doesn't do it!
    # randnum = random.randint(0, 100)
    random.seed(seed)
    random.shuffle(train_question)
    random.seed(seed)
    random.shuffle(train_class)
    random.seed(seed)
    random.shuffle(train_entity)

    with codecs.open(target_train, 'w', encoding='utf8') as f:
        for x, y, z in zip(tqdm(train_question, 'generate train.tsv'), train_class, train_entity):
            f.write('%s\t%d\t%s\n' % (x, y, z))

    with codecs.open(target_test, 'w', encoding='utf8') as f:
        for x, y, z in zip(tqdm(test_question, 'generate test.tsv'), test_class, test_entity):
            f.write('%s\t%d\t%s\n' % (x, y, z))


def main():
    parser = argparse.ArgumentParser(description='generate train file and test file for multi task learning')
    parser.add_argument('--strain', help='source train file')
    parser.add_argument('--sdev', help='source dev file')
    parser.add_argument('--train', help='target train file')
    parser.add_argument('--test', help='target test file')
    parser.add_argument('--label', help='label2id file')
    parser.add_argument('--random', help='random seed')
    args = parser.parse_args()
    generate_multi_task_data(args.strain, args.sdev, args.train, args.test, args.label, int(args.random))


if __name__ == '__main__':
    main()
