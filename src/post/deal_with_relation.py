import pickle
import codecs
import numpy as np
import json
import argparse
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def deal_with_relation(results, analysis, number, topn):
    with codecs.open(results, 'r', encoding='utf8') as f:
        probabilities, predict, gold = json.load(f)
        probabilities = np.array(probabilities)
        gold = np.array(gold)
        assert len(probabilities) == len(predict) == len(gold)
        cnt0_5 = 0
        one_probabilities = probabilities[:, 1]
        length = int(len(probabilities) / number)
        cnt = 0

        for i in range(length):
            g = gold[i * number:(i + 1) * number]
            o = one_probabilities[i * number:(i + 1) * number]
            correct = g.argmax()

            if o[correct] > 0.5:
                cnt += 1

        print(cnt / length)


# deal with type classify json results
def deal_with_type_classify(results, analysis, number, topn):
    with codecs.open(results, 'r', encoding='utf8') as f:
        probabilities, predict, gold = json.load(f)
        probabilities = np.array(probabilities)
        gold = np.array(gold)
        assert len(probabilities) == len(predict) == len(gold)
        predict_set = []
        gold_set = []
        one_probabilities = probabilities[:, 1]
        length = int(len(probabilities) / number)
        cnt = 0

        for i in range(length):
            g = gold[i * number:(i + 1) * number]
            o = one_probabilities[i * number:(i + 1) * number]
            candicate = set(o.argsort()[-topn:])
            correct = g.argmax()

            if correct in candicate:
                cnt += 1

    with codecs.open(analysis, 'a', encoding='utf8') as f:
        f.write('top%d:%.6f\n' % (topn, cnt / length))
        # for index in range(len(predict)):
        #     if predict[index] == 1:
        #         predict_set.append(index)
        #
        #     if gold[index] == 1:
        #         gold_set.append(index)
        #
        # predict_set = set(predict_set)
        # gold_set = set(gold_set)
        # union_set = predict_set & gold_set
        # print('precision:%.6f\nrecall:%.6f\n' % (len(union_set) / len(predict_set), len(union_set) / len(gold_set)))


def main():
    parser = argparse.ArgumentParser(description='deal with type classify results')
    parser.add_argument('-r', '--result', help='type classify result file')
    parser.add_argument('-a', '--analysis', help='analysis results')
    parser.add_argument('-n', '--number', help='number')
    args = parser.parse_args()
    # deal_with_type_classify(args.result, args.analysis, int(args.number), 1)
    # deal_with_type_classify(args.result, args.analysis, int(args.number), 5)
    # deal_with_type_classify(args.result, args.analysis, int(args.number), 10)
    # deal_with_type_classify(args.result, args.analysis, int(args.number), 20)
    # deal_with_type_classify(args.result, args.analysis, int(args.number), 50)
    # deal_with_type_classify(args.result, args.analysis, int(args.number), 100)
    deal_with_relation(args.result, args.analysis, int(args.number), 100)


if __name__ == '__main__':
    main()
