import json
import codecs
import sys
import math
import re
import argparse


json_load = lambda x:json.load(codecs.open(x, 'r', encoding='utf-8'))

def loss_score(prediction):
    pred = prediction[-1]
    if pred == -1:
        s = 0.5
    else:
        s = 1 - (pred[0] + pred[1] + pred[2]) / 3
    return s

def get_cover_prob(question_words, logical):
    question_words = [w for w in question_words if not (w.startswith('ENTITY') or w.startswith('VALUE'))]
    question_words = [w for w in question_words if w not in ['|||']]
    logical_words = [w for w in logical.split(' ') if w.startswith('mso:') or w.startswith('r-mso:') or w.startswith('dev:')]
    logical_words = [w.lstrip('msor-dev') for w in logical_words]
    logical_words = [w.lstrip(':') for w in logical_words]
    logical_words = [re.split(r'[\._]', w) for w in logical_words]
    logical_words = [ww for w in logical_words for ww in w]
    cnt = 0
    for w in question_words:
        if w in logical_words:
            cnt += 1
    return cnt

def return_to_raw(logical, type, id):
    logical = logical.split(' ')
    if type == 'multi-turn-predicate':
        assert '|||' in logical
        pos = logical.index('|||')
        logical = logical[:pos] + [')'] + logical[pos:]
    return ' '.join(logical)

def score(s1, s2, s3, s4, type, opt):
    tmp_s2 = 1
    for x in s2:
        tmp_s2 *= x
    cnt2 = len(s2)
    s2 = tmp_s2
    #s2 = s2/cnt2 if cnt2 != 0 else 1
    s2 = math.pow(s2, 1/cnt2) if cnt2 != 0 else 1
    # if type in ['aggregation']:
    #     s1 = math.log(s1)
    #     s2 = math.log(s2)
    s1 = -100+s1 if s1 < 0.5 else s1
    # if s2 > 0.8:
    #     s2 = 1
    para = opt.todo1to3
    if type == 'aggregation':
        para = opt.todo1to3_aggregation
    elif type == 'single-relation':
        para = opt.todo1to3_singlerelation
    elif type == 'superlative0':
        para = opt.todo1to3_superlative0
    elif type == 'multi-choice':
        para = opt.todo1to3_multichoice
    s = s1 + s2 * para
    para2 = opt.cover2todo13
    s = s/(1 + para) + para2 * s3
    para3 = opt.loss2others
    if type == 'aggregation':
        para3 = opt.
    s = s/(1 + para2) + s4 * para3
    return s, tmp_s2

def process(sample, opt):
    type_pred = sample['type_pred']
    predlist = sample['logical_pred']
    if type_pred in ['multi-turn-entity']:
        q1list = sample['logical_pred_0']
        q2list = sample['logical_pred_1']
        for q1 in q1list:
            for q2 in q2list:
                LP = q1[0] + ' ||| ' + q2[0]
                S = q1[1] * q2[1]
                ER = q1[2] + q2[2]
                LS = -1
                predlist.append([LP, S, ER, LS])
        predlist_dict={}
        for lp,s,er,ls in predlist:
            if not lp in predlist_dict:
                predlist_dict[lp] = [0, er, ls]
            predlist_dict[lp][0] += s
        predlist = [[k, predlist_dict[k][0], predlist_dict[k][1], predlist_dict[k][2]] for k in predlist_dict]
    # 结合TODO1和TODO3
    target, max_score = -1, -100000000000
    for index, pred in enumerate(predlist):
        s1 = pred[1]
        s2 = [er[3] for er in pred[2]]
        s3 = get_cover_prob(sample['question_pre_pattern'], pred[0])
        s4 = loss_score(pred)
        final_score, s2 = score(s1, s2, s3, s4, type_pred, opt)
        predlist[index].append(s2)
        if final_score > max_score:
            max_score = final_score
            target = index
    logical_final_pred = predlist[target][0]
    return logical_final_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score_merge_dev.py')
    parser.add_argument('-input_path', default='', help="""input timestep2-merge in json""")
    parser.add_argument('-result_path', default='', help="""output in txt""")
    parser.add_argument('-todo1to3', type=float, help="""TODO1 : TODO3""")
    parser.add_argument('-todo1to3_aggregation', type=float, help="""TODO1 : TODO3 for aggregation""")
    parser.add_argument('-todo1to3_singlerelation', type=float, help="""TODO1 : TODO3 for single-relation""")
    parser.add_argument('-todo1to3_superlative0', type=float, help="""TODO1 : TODO3 for superlative0""")
    parser.add_argument('-todo1to3_multichoice', type=float, help="""TODO1 : TODO3 for multi-choice""")
    parser.add_argument('-cover2todo13', type=float, help="""cover-score : (TODO1 & TODO3)""")
    parser.add_argument('-loss2others', type=float, help="""loss-score : [cover-score & (TODO1 & TODO3)]""")
    parser.add_argument('-loss2others_aggregation', type=float,
                        help="""loss-score : [cover-score & (TODO1 & TODO3)] for aggregation""")
    opt = parser.parse_args()

    samples = json_load(opt.input_path)
    filename = opt.result_path
    for index, sample in enumerate(samples):
        type = sample['type_pred']
        id = sample['id']
        logical_pred = process(sample, opt)

        logical_pred = return_to_raw(logical_pred, type, id)
        entity_pred = sample['entity_pred']
        print_entity_pred = []
        for i,en in enumerate(entity_pred):
            txt, ty, rg, q = en['text'], en['type'], en['range'], en['q']
            if q == '@Q2':
                pos = sample['question'].index('|||')
                rg[0] -= pos + 1
            if rg[1] == -1:
                rg[0] = -1
            txt = [txt, '(' + ty + ')', rg]
            if q != None and '|||' in sample['question']:
                txt += [q]
            print_entity_pred.append(txt)
            tmp_logical = logical_pred.split(' ')
            for j in range(tmp_logical.count(en['text']) - 1):
                print_entity_pred.append(txt)
        if sample['type_pred'] in ['superlative1', 'superlative2', 'superlative3', 'comparative']:
            pos = 8
            if sample['type_pred'] == 'superlative3':
                pos = 15
            lgclist = logical_pred.split(' ')
            txt = [lgclist[pos], '(type)']
            words = lgclist[pos].split('.')
            words = words[-1].split('_')
            pos = -1
            for i,w in enumerate(sample['question']):
                if i <= len(sample['question']) - len(words):
                    is_equal = True
                    for j in range(len(words)):
                        if sample['question'][i + j] != words[j]:
                            is_equal = False
                            break
                    if is_equal:
                        pos = i
                        break
            pose = -1
            if pos >= 0:
                pose = pos + len(words) - 1
            txt += [[pos, pose]]
            print_entity_pred.append(txt)
        if '|||' not in sample['question']:
            for i,pep in enumerate(print_entity_pred):
                if pep[2][0] == -1:
                    print_entity_pred[i][2][0] += 1000
                elif sample['type_pred'].startswith('superlative') and pep[1] == '(value)':
                    print_entity_pred[i][2][0] += 1000
                elif sample['type_pred'] == 'comparative' and pep[1] == '(type)':
                    print_entity_pred[i][2][0] += 1000
            print_entity_pred.sort(key=lambda x:(x[2][0], x[1][1]))
            for i,pep in enumerate(print_entity_pred):
                if pep[2][0] > 500:
                    print_entity_pred[i][2][0] -= 1000
        for i,pep in enumerate(print_entity_pred):
            txt = [pep[0], pep[1]]
            rg = pep[2]
            txt.append('[' + str(rg[0]) + ',' + str(rg[1]) + ']')
            if len(pep) > 3:
                txt.append(pep[3])
            txt = ' '.join(txt)
            print_entity_pred[i] = txt
        print_entity_pred = ' ||| '.join(print_entity_pred)

        file = 'results/' + filename + '.all'
        ttt = type
        if type.startswith('superlative'):
            ttt = 'superlative'
        with open(file, 'a', encoding='utf-8') as f:
            f.write('<question id=' + str(id) + '>\t' + ' '.join(sample['question']) + '\n')
            f.write('<logical form id=' + str(id) + '>\t' + logical_pred + '\n')
            f.write('<parameters id=' + str(id) + '>\t' + print_entity_pred + '\n')
            f.write('<question type id=' + str(id) + '>\t' + ttt + '\n')
            f.write('==================================================\n')
