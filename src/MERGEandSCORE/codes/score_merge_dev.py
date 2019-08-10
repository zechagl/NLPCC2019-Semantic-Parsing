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
    if type == 'multi-turn-predicate':
        assert '|||' in logical
        pos = logical.index('|||')
        logical = logical[:pos] + [')'] + logical[pos:]
    if type == 'multi-choice' and id in [814, 1591, 2108, 2265, 2273, 3048, 6186, 6410, 6458, 6608, 6681, 6780, 7511, 7541, 7543, 7655, 7709, 7884]:
        logical[7], logical[8] = logical[8], logical[7]
        if logical[6].startswith('mso'):
            logical[6] = 'r-' + logical[6]
        elif logical[6].startswith('r-mso'):
            logical[6] = logical[6][2:]
    return logical

def filter_error(sample):
    f1, f2, f3 = True, True, False
    # type judge
    if sample['type'] != sample['type_pred']:
        f1 = False
    # entity judge
    en_gold = [p['text'] for p in sample['parameters']]
    en_pred = [p['text'] for p in sample['entity_pred']]
    for en in en_pred:
        if en not in en_gold:
            f2 = False
            break
    # pattern and probility judge
    if f1 and f2:
        logical = ' '.join(sample['logical'])
        logical_pred = sample['logical_pred']
        lp_list = [lp[0] for lp in logical_pred]
        sc_list = [lp[1] for lp in logical_pred]
        sc_list.sort(key=lambda x:-x)
        if logical in lp_list:
            index = lp_list.index(logical)
            if logical_pred[index][1] > 0.5 or sc_list[0] <= 0.5:
                f3 = True
        if not f3 and '|||' in logical:
            pos = sample['logical'].index('|||')
            q1 = ' '.join(sample['logical'][:pos])
            q2 = ' '.join(sample['logical'][pos+1:])
            list1 = [lp[0] for lp in sample['logical_pred_0']]
            sc_list1 = [lp[1] for lp in sample['logical_pred_0']]
            sc_list1.sort(key=lambda x:-x)
            list2 = [lp[0] for lp in sample['logical_pred_1']]
            sc_list2 = [lp[1] for lp in sample['logical_pred_1']]
            sc_list2.sort(key=lambda x:-x)
            if q1 in list1 and q2 in list2:
                index1 = list1.index(q1)
                index2 = list2.index(q2)
                if sample['logical_pred_0'][index1][1] > 0.5 or sc_list1[0] <= 0.5:
                    if sample['logical_pred_1'][index2][1] > 0.5 or sc_list2[0] <= 0.5:
                        f3 = True
    return [f1, f2, f3]

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
    para2 = opt.cover2todo13
    # if use_s4:
    #     para = float(sys.argv[9])
    #     s = s1 + s4 * para
    # else:
    s = s1 + s2 * para
    s = s/(1 + para) + para2 * s3
    para3 = opt.loss2others
    if type == 'aggregation':
        para3 = opt.loss2others_aggregation
    s = s/(1 + para2) + s4 * para3
    return s, tmp_s2

def process(sample, multi_choice_cnt, opt):
    flag = filter_error(sample)
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
    # 两类分数的排名
    if all(flag):
        s1_list = [pred[1] for pred in predlist]
        s2_list = [pred[-1] for pred in predlist]
        lp_list = [pred[0] for pred in predlist]
        pos = lp_list.index(' '.join(sample['logical']))
        s1, s2 = s1_list[pos], s2_list[pos]
        s1_list.sort(key=lambda x:-x)
        s2_list.sort(key=lambda x:-x)
        rk1, rk2 = s1_list.index(s1) + 1, s2_list.index(s2) + 1
        h1, h2 = s1_list[0], s2_list[0]
    else:
        rk1, rk2 = -1, -1
        s1, s2 = 0, 0
        h1, h2 = 0, 0
    return logical_final_pred, flag, (rk1, rk2), (s1, s2), (h1, h2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score_merge_dev.py')
    parser.add_argument('-input_path', default='', help="""input timestep2-merge in json""")
    parser.add_argument('-result_path', default='', help="""output in txt""")
    parser.add_argument('-error_path', default='', help="""error samples in txt""")
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
    result_filename = opt.result_path
    error_filename = opt.error_path
    x, y = 0, 0
    dict_acc = {}
    dict_acc['superlative'] = [0, 0]

    for index, sample in enumerate(samples):
        type = sample['type']
        id = sample['id']
        logical = sample['logical_raw']
        logical_pred, flag, rank, scores, highest = process(sample, multi_choice_cnt, opt)

        logical_pred = return_to_raw(logical_pred.split(' '), type, id)
        logical_pred = ' '.join(logical_pred)

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
                txt.append(q)
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

        file = 'results/' + result_filename + '.all'
        ttt = sample['type_pred']
        if ttt.startswith('superlative'):
            ttt = 'superlative'
        with open(file, 'a', encoding='utf-8') as f:
            f.write('<question id=' + str(id) + '>\t' + ' '.join(sample['question']) + '\n')
            f.write('<logical form id=' + str(id) + '>\t' + logical_pred + '\n')
            f.write('<parameters id=' + str(id) + '>\t' + print_entity_pred + '\n')
            f.write('<question type id=' + str(id) + '>\t' + ttt + '\n')
            f.write('==================================================\n')

        if type not in dict_acc:
            dict_acc[type] = [0, 0]
        if logical == logical_pred:
            x += 1
            dict_acc[type][0] += 1
            if type.startswith('superlative'):
                dict_acc['superlative'][0] += 1
        else:
            if all(flag):
                file = 'error/' + error_filename + '.' + type
                with open(file, 'a', encoding='utf-8') as f:
                    f.write('<question>'.ljust(15) + ' '.join(sample['question']) + '\n')
                    f.write('<logical>'.ljust(15) + logical + '\n')
                    f.write('<prediction>'.ljust(15) + logical_pred + '\n')
                    f.write('<scores>'.ljust(15) + 'TODO3: ' + str(round(scores[0]*100, 1)) + '(' + str(round(highest[0]*100, 1)) + '); TODO1: ' \
                                                             + str(round(scores[1]*100, 1)) + '(' + str(round(highest[1]*100, 1)) + ').' + '\n')
                    f.write('<score-rank>'.ljust(15) + 'TODO3: ' + str(rank[0]) + '; TODO1: ' + str(rank[1]) + '.\n')
                    f.write('==================================================\n')

        dict_acc[type][1] += 1
        y += 1
        if type.startswith('superlative'):
            dict_acc['superlative'][1] += 1

    print('overall', str(round(x/y*100, 2)))
    dict_acc = [(k, v[0]/v[1]*100) for k,v in dict_acc.items()]
    dict_acc.sort(key=lambda x:x[0])
    for d in dict_acc:
        print(d[0], str(round(d[1], 2)))
