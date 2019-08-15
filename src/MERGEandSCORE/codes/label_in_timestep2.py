import json
import codecs
from try_ts import get_type_pattern, check_pattern
import argparse


json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
list = ['first', 'last', 'most', 'top', 'recent', '1st', 'new', 'recently', 'second']

def data_load(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        data = fin.read()
    data = data.split('==================================================\n')
    data = data[0:-1]
    data = [d.strip().split('\n') for d in data]
    data = [[s.split('\t') for s in sample] for sample in data]
    return data

def labelfordata(input, labels, mode):
    '''验证集和测试集的 预测15类 type_pred'''
    type_dict = json_load('index/class2id.json')
    type_dict = dict([(value, key) for key, value in type_dict.items()])
    label_dict = json_load('index/el2id.json')
    output = []
    for n in range(9000):
        sample = input[n]
        type = type_dict[labels[5][n]]
        sample['type_pred'] = type
        sample['type_pred_pattern'] = get_type_pattern(type)
        if mode == 'dev':
            sample['type_pred_pattern_correct'] = check_pattern(sample['type_pred_pattern'], \
                                                                sample['logical'], \
                                                                sample['parameters'])
        else:
            sample['type_pred_pattern_correct'] = None

        label, ref, prob = labels[1][n], labels[3][n], labels[2][n]
        label, ref, prob = [l for l in label if l != 0], [l for l in ref if l != 0], \
                           [[ll for ll in l] for l in prob if l != 0]
        label, ref, prob = label[1:-1], ref[1:-1], prob[1:-1]
        lab, prb = [], []
        ### 生成 entity_pred
        for i in range(len(label)):
            l = label[i]
            r = ref[i]
            if r != 4:
                lab.append(label_dict[str(l)])
                prb.append([prob[i]])
            else:
                prb[-1].append(prob[i])
        entity_pred = []
        qu = sample['question']
        if mode == 'test':
            if len(qu) > 1:
                qu = qu[0] + ['|||'] + qu[1]
            else:
                qu = qu[0]
            sample['question'] = qu
        dil = len(qu)
        if '|||' in qu:
            dil = qu.index('|||')
        i = 0
        while i < len(qu):
            lb = lab[i]
            if lb == 'b':
                dicts = {}
                dicts['text'] = qu[i]
                dicts['type'] = 'entity'
                if dil == len(lab):
                    dicts['range'] = [i]
                    dicts['q'] = None
                elif i < dil:
                    dicts['range'] = [i]
                    dicts['q'] = '@Q1'
                else:
                    dicts['range'] = [i - dil - 1]
                    dicts['q'] = '@Q2'
                if i == len(lab) - 1 or lab[i + 1] != 'm':
                    if i < dil:
                        dicts['range'].append(i)
                    else:
                        dicts['range'].append(i - dil - 1)
                i += 1
                while i < len(lab) and lab[i] == 'm':
                    #print(qu, lab)
                    dicts['text'] += '_' + qu[i]
                    if i == len(lab) - 1 or lab[i + 1] != 'm':
                        if i < dil:
                            dicts['range'].append(i)
                        else:
                            dicts['range'].append(i - dil - 1)
                    i += 1
                entity_pred.append(dicts)
            else:
                i += 1
        if lab[i] == 'b':
            if type.startswith('superlative'):
                flag = False
                for w in sample['question']:
                    if w in list or w.count('min')+w.count('max') > 0 or w.endswith('est'):
                        flag = True
                        break
                if flag:
                    dicts = {}
                    dicts['text'] = '1'
                    dicts['type'] = 'value'
                    dicts['range'] = [-1,-1]
                    dicts['q'] = None
                    entity_pred.append(dicts)
            else:
                type_not_superlative += 1
        sample['entity_pred'] = entity_pred
        output.append(sample)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='label_in_timestep2.py')
    parser.add_argument('-mode', default='dev',choices=['dev', 'test'])
    parser.add_argument('-input_path', default='', help="""input timestep2-v1 in json""")
    parser.add_argument('-label_path', default='', help="""input label result in json""")
    parser.add_argument('-result_path', default='', help="""output timestep2-v2 in json""")
    opt = parser.parse_args()

    inputs = json_load(opt.input_path)
    labels = json_load(opt.label_path)
    data = labelfordata(inputs, labels, opt.mode)
    json_dump(data, opt.result_path)
