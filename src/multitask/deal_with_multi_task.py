import codecs
import numpy as np
import json
import argparse
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


'''
get entity according to label list and word list
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


'''
deal with results of multi-task model
'''
def deal_with_multi_task(results, tokens, entity_label2id, sketch_label2id, analysis):
    with codecs.open(results, 'r', encoding='utf8') as f:
        entity_labels, entity_predict, entity_probabilities, entity_reference, sketch_labels, sketch_predict, sketch_probabilities = json.load(f)
    # deal_with_entity_labeling(entity_labels, entity_predict, entity_probabilities, entity_reference, token, entity_label2id, entity_analysis)
    # deal_with_entity_labeling(ori_labels, des_labels, probabilities, reference, tokens, label2id, predict_entities):

    # sketch classification
    with codecs.open(sketch_label2id, 'r', encoding='utf8') as f:
        d = json.load(f)
        sketch_id2label = dict([(number, label) for label, number in d.items()])

    labels = list(d.keys())
    cm = confusion_matrix(sketch_labels, sketch_predict)
    cm = np.array(cm)

    with codecs.open(analysis, 'w', encoding='utf8') as f:
        f.write('Sketch Classification:\nConfusion Matrix\n')
        for label in labels:
            f.write(',%s' % label)

        f.write('\n')

        for i in range(len(cm)):
            f.write(labels[i])
            for j in range(len(cm[i])):
                f.write(',%d' % cm[i][j])

            f.write('\n')

        f1_scores = f1_score(sketch_labels, sketch_predict, average=None)
        f.write('\n\n\nErr_s\n')

        for label, f1 in zip(labels, f1_scores):
            f.write('%s:%.6f\n' % (label, 100 * (1 - f1)))

        f.write('\n\n\nOverall\n')
        f.write('acc:%.6f\n' % accuracy_score(sketch_labels, sketch_predict))
        f.write('f1:%.6f\n' % f1_score(sketch_labels, sketch_predict, average='weighted'))
        f.write('precision:%.6f\n' % precision_score(sketch_labels, sketch_predict, average='weighted'))
        f.write('recall:%.6f\n' % recall_score(sketch_labels, sketch_predict, average='weighted'))

    with codecs.open(analysis + '_sketch.json', 'w', encoding='utf8') as f:
        sketches = []

        for number in sketch_predict:
            sketches.append(sketch_id2label[number])

        json.dump(sketches, f, indent=2)

    # entity labeling
    questions = []
    entities = []
    entity_sketch_dict = dict([(sketch, []) for sketch in sketch_id2label.values()])
    multitask_sketch_dict = dict([(sketch, []) for sketch in sketch_id2label.values()])

    with codecs.open(tokens, 'r', encoding='utf8') as f:
        for line in f:
            questions.append(line.strip())

    with codecs.open(entity_label2id, 'r', encoding='utf8') as f:
        d = json.load(f)
        d = dict([(value, key) for key, value in d.items()])
        d[0] = '0'

    ori_labels = np.array(entity_labels)
    des_labels = np.array(entity_predict)
    reference = np.array(entity_reference)
    assert len(ori_labels) == len(des_labels) == len(questions) == len(reference)
    entity_cnt = 0
    multi_task_cnt = 0

    for ori, des, que, pro, ref, sla, spr in zip(ori_labels, des_labels, questions, entity_probabilities, reference, sketch_labels, sketch_predict):
        que = que.strip().split()
        ori = list(map(lambda x: d[x], ori))
        des = list(map(lambda x: d[x], des))
        ref = list(map(lambda x: d[x], ref))
        que, ori, des, pro = deal_with_x(que, ori, des, pro, ref)
        gold = get_entity(que, ori)
        predict = get_entity(que, des)
        entities.append(predict)
        gold = set(gold)
        predict = set(predict)
        multitask_sketch_dict[sketch_id2label[int(sla)]].append(1 if (gold == predict and sla == spr) else 0)
        entity_sketch_dict[sketch_id2label[int(sla)]].append(1 if (gold == predict) else 0)

        if gold == predict:
            entity_cnt += 1

            if sla == spr:
                multi_task_cnt += 1

    with codecs.open(analysis, 'a', encoding='utf8') as f:
        f.write('\n\n\n\nEntity Labeling:\nErr_e\n')

        for sketch, label_list in entity_sketch_dict.items():
            f.write('%s:%.6f\n' % (sketch, 100 * (1 - sum(label_list) / len(label_list))))

        f.write('overall:%.6f' % (100 * (1 - entity_cnt / 9000)))
        f.write('\n\n\n\nErr_m\n')

        for sketch, label_list in multitask_sketch_dict.items():
            f.write('%s:%.6f\n' % (sketch, 100 * (1 - sum(label_list) / len(label_list))))

        f.write('overall:%.6f' % (100 * (1 - multi_task_cnt / 9000)))

    with codecs.open(analysis + '_entity.json', 'w', encoding='utf8') as f:
        # assert len(entities) == 9000
        json.dump(entities, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='deal with results of multi-task model')
    parser.add_argument('-r', '--result', help='multi task result file')
    parser.add_argument('-t', '--token', help='entity labeling token file')
    parser.add_argument('-e', '--entity_label2id', help='entity labeling label2id dict')
    parser.add_argument('-s', '--sketch_label2id', help='sketch classification label2id dict')
    parser.add_argument('-a', '--analysis', help='type classify analysis results')
    args = parser.parse_args()
    deal_with_multi_task(args.result, args.token, args.entity_label2id, args.sketch_label2id, args.analysis)


if __name__ == '__main__':
    main()
