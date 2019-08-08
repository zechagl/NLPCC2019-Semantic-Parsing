import pickle
import codecs
import numpy as np
import json
import argparse
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# deal with type classify json results
def deal_with_type_classify(results, label2id, analysis):
    questions = []

    with codecs.open(results, 'r', encoding='utf8') as f:
        probabilities, predict, gold = json.load(f)

    with codecs.open(label2id, 'r', encoding='utf8') as f:
        d = json.load(f)

    labels = list(d.keys())
    cm = confusion_matrix(gold, predict)
    m = np.array(cm)

    with codecs.open(analysis, 'w', encoding='utf8') as f:
        for label in labels:
            f.write(',%s' % label)

        f.write('\n')

        for i in range(len(m)):
            f.write(labels[i])
            for j in range(len(m[i])):
                f.write(',%d' % m[i][j])

            f.write('\n')

        f.write('acc:%.6f\n' % f1_score(gold, predict, average='micro'))
        f.write('macro-f1:%.6f\n' % f1_score(gold, predict, average='macro'))
        f.write('macro-precision:%.6f\n' % precision_score(gold, predict, average='macro'))
        f.write('macro-recall:%.6f\n' % recall_score(gold, predict, average='macro'))


def main():
    parser = argparse.ArgumentParser(description='deal with type classify results')
    parser.add_argument('-r', '--result', help='type classify result file')
    parser.add_argument('-l', '--label2id', help='file to store label2id dict')
    parser.add_argument('-a', '--analysis', help='analysis results')
    args = parser.parse_args()
    deal_with_type_classify(args.result, args.label2id, args.analysis)


if __name__ == '__main__':
    main()
