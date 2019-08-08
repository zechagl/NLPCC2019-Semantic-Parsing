import codecs
import numpy as np
import json
import argparse
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


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
    #assert predict[0] == 'o' and predict[len(words) + 1] == 'o' and predict[len(words) + 2] == '0'
    #assert gold[0] == 'o' and gold[len(words) + 1] == 'o' and gold[len(words) + 2] == '0'
    #assert reference[0] == 'o' and reference[len(words) + 1] == 'o' and reference[len(words) + 2] == '0'
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


def deal_with_tr(ori_labels, des_labels, probabilities, reference, tokens, label2id, class_labels, class_label2id, analysis, class_predict):
    questions = []

    with codecs.open(tokens, 'r', encoding='utf8') as f:
        for line in f:
            questions.append(line.strip())

    with codecs.open(label2id, 'r', encoding='utf8') as f:
        d = json.load(f)
        d = dict([(value, key) for key, value in d.items()])
        d[0] = '0'

    with codecs.open(class_label2id, 'r', encoding='utf8') as f:
        c = json.load(f)
        c = dict([(value, key) for key, value in c.items()])
        class_dict = dict([(class_name, []) for class_name in c.values()])

    ori_labels = np.array(ori_labels)
    des_labels = np.array(des_labels)
    reference = np.array(reference)
    pred_num = 0
    assert len(ori_labels) == len(des_labels) == len(probabilities) == len(questions) == len(reference)

    with codecs.open(analysis, 'w', encoding='utf8') as f:
        for ori, des, que, pro, ref, cla, cpr in zip(ori_labels, des_labels, questions, probabilities, reference, class_labels, class_predict):
            que = que.strip().split()
            ori = list(map(lambda x: d[x], ori))
            des = list(map(lambda x: d[x], des))
            ref = list(map(lambda x: d[x], ref))
            que, ori, des, pro = deal_with_x(que, ori, des, pro, ref)
            gold = get_entity(que, ori)
            predict = get_entity(que, des)
            gold = set(gold)
            predict = set(predict)
            if gold == predict and cla == cpr:
                pred_num += 1

            class_dict[c[int(cla)]].append(1 if (gold == predict and cla == cpr) else 0)

            if gold != predict:
                f.write('%s\n' % ' '.join(que))
                f.write('%s\n' % c[int(cla)])

                for gold_item in gold:
                    f.write('gold:%s\n' % gold_item)

                for predict_item in predict:
                    f.write('predict:%s\n' % predict_item)

                f.write('\n')

        for key, value in class_dict.items():
            f.write('%s:%.6f\n' % (key, 1 - sum(value) / len(value)))
        f.write('%.6f' % (pred_num / 9000))


# deal with multi task json file
def deal_with_multi_task(results, token, entity_label2id, entity_analysis, class_label2id, class_analysis):
    with codecs.open(results, 'r', encoding='utf8') as f:
        entity_labels, entity_predict, entity_probabilities, entity_reference, class_labels, class_predict, class_probabilities = json.load(f)

    deal_with_tr(entity_labels, entity_predict, entity_probabilities, entity_reference, token, entity_label2id, class_labels, class_label2id, entity_analysis, class_predict)


def main():
    parser = argparse.ArgumentParser(description='deal with entity labeling results')
    parser.add_argument('-r', '--result', help='multi task result file')
    parser.add_argument('-t', '--token', help='entity labeling token file')
    parser.add_argument('-l', '--elabel2id', help='file to store entity labeling label2id dict')
    parser.add_argument('-a', '--eanalysis', help='entity labeling analysis results')
    parser.add_argument('-d', '--tlabel2id', help='file to store type classify label2id dict')
    parser.add_argument('-s', '--tanalysis', help='type classify analysis results')
    args = parser.parse_args()
    deal_with_multi_task(args.result, args.token, args.elabel2id, args.eanalysis, args.tlabel2id, args.tanalysis)


if __name__ == '__main__':
    main()
