import json
import codecs
import argparse
from tqdm import tqdm

from src.score.logic_form import get_logic
from src.score.score_merge import return_to_raw


json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def score(r, e, d, f):
    try:
        s = d[str(f)][e][r]
    except:
        f = 1 - f
        if r.startswith('mso:'):
            r = 'r-' + r
        elif r.startswith('r-mso:'):
            r = r.lstrip('r-')
        try:
            s = d[str(f)][e][r]
        except:
            s = 0.000000000000001
    return s

def process(data, prediction, lgp_dict, dict_full):
    pred_type = prediction[0]['type']
    predlist = prediction[0]['pred']
    for i, p in enumerate(predlist):
        try:
            predlist[i] = [lgp_dict[pred_type][p[0]].split(' '), p[1]]
        except:
            predlist[i] = [get_logic(pred_type, p[0].split(' ')).split(' '), p[1]]
    q1list, q2list = [], []
    if len(prediction) > 1:
        q1list = prediction[1]['pred']
        q2list = prediction[2]['pred']
        tmp1, tmp2 = q1list[0], q2list[0]
        for i, q in enumerate(q1list):
            try:
                q1list[i] = [lgp_dict[pred_type + '_0'][q[0]].split(' '), q[1]]
            except:
                Q = get_logic(pred_type, q[0].split(' ') + ['|||'] + tmp2[0].split(' ')).split(' ')
                pos = Q.index('|||')
                Q = Q[0:pos]
                q1list[i] = [Q, q[1]]
        for i, q in enumerate(q2list):
            try:
                q2list[i] = [lgp_dict[pred_type + '_1'][q[0]].split(' '), q[1]]
            except:
                Q = get_logic(pred_type, tmp1[0].split(' ') + ['|||'] + q[0].split(' ')).split(' ')
                pos = Q.index('|||')
                Q = Q[pos+1:]
                q2list[i] = [Q, q[1]]
    # 筛除 logic pattern 中 entity/value 不合法的项
    question_pred_pattern = ' '.join(data['question_pre_pattern'])
    question_pred_pattern = question_pred_pattern.split(' ')
    entity_names = [w for w in question_pred_pattern if w.startswith('ENTITY') or w.startswith('VALUE')]
    entity_pred = data['entity_pred']
    minus_one = [en for en in entity_pred if en['range'][0] == -1]
    if len(minus_one) > 0:
        entity_names.append('VALUE' + str(len(entity_names) + 1))
    pse_predlist = []
    for pred in predlist:
        flag = True
        for w in pred[0]:
            if w.startswith('ENTITY') or w.startswith('VALUE'):
                if w not in entity_names:
                    flag = False
                    break
        if flag:
            pse_predlist.append(pred)
    predlist = pse_predlist
    pseq1, pseq2 = [], []
    for q in q1list:
        flag = True
        for w in q[0]:
            if w.startswith('ENTITY') or w.startswith('VALUE'):
                if w not in entity_names:
                    flag = False
                    break
        if flag:
            pseq1.append(q)
    q1list = pseq1
    for q in q2list:
        flag = True
        for w in q[0]:
            if w.startswith('ENTITY') or w.startswith('VALUE'):
                if w not in entity_names:
                    flag = False
                    break
        if flag:
            pseq2.append(q)
    q2list = pseq2
    # 生成 entity/value 字典
    for i,en in enumerate(entity_pred):
        if en['range'][0] == -1:
            entity_pred[i]['range'][0] = 1000
        if en['q'] == '@Q2':
            entity_pred[i]['range'][0] += data['question'].index('|||') + 1
    entity_pred.sort(key=lambda x : x['range'][0])
    entity_dict = {}
    for i, n in enumerate(entity_names):
        entity_dict[n] = entity_pred[i]['text']
    # 获取 TODO1 分数并得到加权分，进一步得到加权分最高者
    er_scores = []
    for i, pred in enumerate(predlist):
        ers = []
        logical_pred_pattern = pred[0]
        for j,w in enumerate(logical_pred_pattern):
            if w.startswith('mso:') or w.startswith('r-mso:') or w.startswith('dev:'):
                if j + 2 < len(logical_pred_pattern):
                    if logical_pred_pattern[j+1] in entity_names:
                        en = entity_dict[logical_pred_pattern[j+1]]
                        scr = score(w, en, dict_full, 0)
                        ers.append([en, w, 0, scr])
                    elif logical_pred_pattern[j+2] in entity_names:
                        en = entity_dict[logical_pred_pattern[j+2]]
                        scr = score(w, en, dict_full, 1)
                        ers.append([en, w, 1, scr])
        er_scores.append(ers)
        for j,w in enumerate(logical_pred_pattern):
            if w in entity_names:
                predlist[i][0][j] = entity_dict[w]
        predlist[i][0] = ' '.join(predlist[i][0])
    erqscore1, erqscore2 = [], []
    for i, pred in enumerate(q1list):
        logical_pred_pattern = pred[0]
        ers = []
        for j,w in enumerate(logical_pred_pattern):
            if w.startswith('mso:') or w.startswith('r-mso:') or w.startswith('dev:'):
                if j + 2 < len(logical_pred_pattern):
                    if logical_pred_pattern[j+1] in entity_names:
                        en = entity_dict[logical_pred_pattern[j+1]]
                        scr = score(w, en, dict_full, 0)
                        ers.append([en, w, 0, scr])
                    elif logical_pred_pattern[j+2] in entity_names:
                        en = entity_dict[logical_pred_pattern[j+2]]
                        scr = score(w, en, dict_full, 1)
                        ers.append([en, w, 1, scr])
        erqscore1.append(ers)
        for j,w in enumerate(logical_pred_pattern):
            if w in entity_names:
                q1list[i][0][j] = entity_dict[w]
        q1list[i][0] = ' '.join(q1list[i][0])
    for i, pred in enumerate(q2list):
        logical_pred_pattern = pred[0]
        ers = []
        for j,w in enumerate(logical_pred_pattern):
            if w.startswith('mso:') or w.startswith('r-mso:') or w.startswith('dev:'):
                if j + 2 < len(logical_pred_pattern):
                    if logical_pred_pattern[j+1] in entity_names:
                        en = entity_dict[logical_pred_pattern[j+1]]
                        scr = score(w, en, dict_full, 0)
                        ers.append([en, w, 0, scr])
                    elif logical_pred_pattern[j+2] in entity_names:
                        en = entity_dict[logical_pred_pattern[j+2]]
                        scr = score(w, en, dict_full, 1)
                        ers.append([en, w, 1, scr])
        erqscore2.append(ers)
        for j,w in enumerate(logical_pred_pattern):
            if w in entity_names:
                q2list[i][0][j] = entity_dict[w]
        q2list[i][0] = ' '.join(q2list[i][0])
    # 生成最终 logical
    rt_dict = {}
    predlist = [pred + [er_scores[i]] for i,pred in enumerate(predlist)]
    rt_dict['logical_pred'] = predlist
    if len(q1list) > 0:
        q1list = [pred + [erqscore1[i]] for i,pred in enumerate(q1list)]
        rt_dict['logical_pred_0'] = q1list
    if len(q2list) > 0:
        q2list = [pred + [erqscore2[i]] for i,pred in enumerate(q2list)]
        rt_dict['logical_pred_1'] = q2list
    if len(predlist) == 0:
        print(' '.join(data['question_pre_pattern']))
    return rt_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge.py')
    parser.add_argument('-mode', required=True, choices=['dev', 'test'])
    parser.add_argument('-input_path', required=True, help="""input timestep2-v3 in json""")
    parser.add_argument('-pred_path', required=True, help="""input logical form prediction in json""")
    parser.add_argument('-result_path', required=True, help="""output merge in json""")
    parser.add_argument('-loss_path', required=True, default=[], nargs='+', help="""loss results in json""")
    parser.add_argument('-index_path', required=True, default=[], nargs='+', 
                        help="""index files (score of entites and predicates) in json""")
    parser.add_argument('-qu2logical', required=True, help="""dictionary of question : patter in json""")
    # parser.add_argument('-gold_logical_form', default='MSParS.dev', help="""raw file of development set""")
    opt = parser.parse_args()

    print('Loading data and index files ......')

    data = json_load(opt.input_path)
    predictions = json_load(opt.pred_path)
    lgp_dict = json_load(opt.qu2logical)

    # with open(opt.gold_logical_form, 'r', encoding='utf-8') as f:
    #     dev = f.read()
    # dev = dev.split('==================================================\n')
    # dev = dev[0:-1]
    # dev = [d.strip().split('\n') for d in dev]
    # dev = [[s.split('\t') for s in sample] for sample in dev]
    # dev = [sample[1][1] for sample in dev]

    dict_full = {}
    dict_full['0'] = json_load(opt.index_path[0])
    dict_full['1'] = json_load(opt.index_path[1])
    dict_acc = {}
    for index,sample in tqdm(enumerate(data), desc='   - (Loading Pattern Pairs) -   '):
        prediction = predictions[index]
        logical_pred_dict = process(sample, prediction, lgp_dict, dict_full)
        id = sample['id']
        type = sample['type']
        logical_pred = sample['logical']
        # logical_pred = dev[id - 1]

        logical_pred = return_to_raw(logical_pred, type, id, opt.mode)
        logical_pred = ' '.join(logical_pred) 
        # data[index]['logical_raw'] = logical_pred

        for k,v in logical_pred_dict.items():
            data[index][k] = v

    # merge loss results into data
    dict_full = {}
    loss1 = json_load(opt.loss_path[0])
    loss2 = json_load(opt.loss_path[1])
    loss3 = json_load(opt.loss_path[2])
    loss1, loss2, loss3 = loss1[0], loss2[0], loss3[0]
    cnt = 0
    for index, sample in tqdm(enumerate(data), desc='   - (Loading Pointer Loss) -   '):
        logical_pred = sample['logical_pred']
        # if opt.mode == 'dev':
        #     cnt += 1
        for i,lp in enumerate(logical_pred):
            logical_pred[i].append([loss1[cnt], loss2[cnt], loss3[cnt]])
            cnt += 1
        data[index]['logical_pred'] = logical_pred
        if 'logical_pred_0' in sample and 'logical_pred_1' in sample:
            len1 = len(sample['logical_pred_0'])
            len2 = len(sample['logical_pred_1'])
            length = len1 * len2
            v = []
            for i in range(length):
                v.append([loss1[cnt], loss2[cnt], loss3[cnt]])
                cnt += 1
            data[index]['loss_for_split'] = v
    print('Saving results ......')
    json_dump(data, opt.result_path)
