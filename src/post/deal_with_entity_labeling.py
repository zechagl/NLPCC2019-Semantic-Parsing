import pickle
import codecs
import numpy as np
import json
import argparse


'''
get entity according to label list and word list
words is the word list
labels is the label list
'''
def get_entity(words, labels):
    entities = []
    entity = []

    for label, word in zip(labels, words):
        assert label != '0'
        if label == 'o':
            if len(entity) > 0:
                entity = ' '.join(entity)
                entities.append(entity)
                entity = []
        elif label == 'b':
            if len(entity) > 0:
                entity = ' '.join(entity)
                entities.append(entity)
                entity = []

            entity.append(word)
        elif label == 'm':
            entity.append(word)
        else:
            print('err!!!!!!')

    if len(entity) > 0:
        entity = ' '.join(entity)
        entities.append(entity)

    return entities


'''
deal with 'x' label problem due to bert
'''
def deal_with_x(words, gold, predict, probabilities, reference):
#    assert predict[0] == 'o' and predict[len(words) + 1] == 'o' and predict[len(words) + 2] == '0'
#    assert gold[0] == 'o' and gold[len(words) + 1] == 'o' and gold[len(words) + 2] == '0'
#    assert reference[0] == 'o' and reference[len(words) + 1] == 'o' and reference[len(words) + 2] == '0'
    gold = gold[1:len(words) + 1]
    predict = predict[1:len(words) + 1]
    probabilities = probabilities[1:len(words) + 1]
    reference = reference[1:len(words) + 1]
    flag = True

    while flag:
        flag = False

        for index, label in enumerate(reference):
            if label == 'x':
                words[index - 1] = words[index - 1] + (words[index][2:] if words[index].startswith('##') else words[index])
                words = words[:index] + words[index + 1:] if index + 1 < len(words) else words[:index]
                gold = gold[:index] + gold[index + 1:] if index + 1 < len(gold) else gold[:index]
                predict = predict[:index] + predict[index + 1:] if index + 1 < len(predict) else predict[:index]
                probabilities = probabilities[:index] + probabilities[index + 1:] if index + 1 < len(probabilities) else probabilities[:index]
                reference = reference[:index] + reference[index + 1:] if index + 1 < len(reference) else reference[:index]
                flag = True
                break

    return words, gold, predict, probabilities


# deal with entity labeling json results
def deal_with_entity_labeling(results, tokens, label2id, predict_entities):
    questions = []

    with codecs.open(results, 'r', encoding='utf8') as f:
        ori_labels, des_labels, probabilities, reference = json.load(f)

    with codecs.open(tokens, 'r', encoding='utf8') as f:
        for line in f:
            questions.append(line.strip())

    with codecs.open(label2id, 'r', encoding='utf8') as f:
        d = json.load(f)
        d = dict([(value, key) for key, value in d.items()])
        d[0] = '0'

    ori_labels = np.array(ori_labels)
    des_labels = np.array(des_labels)
    reference = np.array(reference)
    assert len(ori_labels) == len(des_labels) == len(probabilities) == len(questions) == len(reference)

    with codecs.open(predict_entities, 'w', encoding='utf8') as f:
        r = []
        err = []
        pre_cnt = 0
        rec_cnt = 0
        total_cnt = 0

        for ori, des, que, pro, ref in zip(ori_labels, des_labels, questions, probabilities, reference):
            que = que.strip().split()
            ori = list(map(lambda x: d[x], ori))
            des = list(map(lambda x: d[x], des))
            ref = list(map(lambda x: d[x], ref))
            que, ori, des, pro = deal_with_x(que, ori, des, pro, ref)
            gold = get_entity(que, ori)
            predict = get_entity(que, des)
            gold = set(gold)
            predict = set(predict)

            if gold == predict:
                pre_cnt += 1
            else:
                err.append(dict({'question': ' '.join(que), 'gold': list(gold), 'predict': list(predict)}))

            rec_cnt += len(gold & predict)
            total_cnt += len(gold)
            f.write('word'.ljust(55, ' '))
            f.write('gold'.ljust(5, ' '))
            f.write('predict'.ljust(8, ' '))
            f.write('padding'.ljust(10, ' '))
            f.write('b'.ljust(10, ' '))
            f.write('m'.ljust(10, ' '))
            f.write('o'.ljust(10, ' '))
            f.write('\n')

            for i, j, k, l in zip(que, ori, des, pro):
                f.write(i.ljust(55, ' '))
                f.write(j.ljust(5, ' '))
                f.write(k.ljust(8, ' '))

                for m in l:
                    f.write('%-10.6f' % m)

                f.write('\n')

            f.write('\n\n')
            r.append(dict({'question': ' '.join(que), 'gold': list(gold), 'predict': list(predict)}))

        if total_cnt != 0:
            f.write('precision:%.6f\nrecall:%.6f\n' % (pre_cnt / len(questions), rec_cnt / total_cnt))

    with codecs.open(predict_entities + '.json', 'w', encoding='utf8') as f:
        json.dump(r, f, indent=2)

    with codecs.open(predict_entities + '_err.json', 'w', encoding='utf8') as f:
        json.dump(err, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='deal with entity labeling results')
    parser.add_argument('-r', '--result', help='entity labeling result file')
    parser.add_argument('-t', '--token', help='entity labeling token file')
    parser.add_argument('-l', '--label2id', help='file to store label2id dict')
    parser.add_argument('-a', '--analysis', help='analysis results')
    args = parser.parse_args()
    deal_with_entity_labeling(args.result, args.token, args.label2id, args.analysis)


if __name__ == '__main__':
    main()
