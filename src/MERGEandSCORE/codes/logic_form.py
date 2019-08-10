import re
from try_ts import get_type_pattern

ignore_token = set(['(', ')', 'and', 'or', 'isa', 'lambda', 'exist', 'equal'])

def get_relation():
    with open('index/relations.txt', 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')
    dict = {}
    for d in data:
        if d.startswith('mso:'):
            key = re.split(r'[\._]', d[4:])
        elif d.startswith('r-mso:'):
            key = ['reverse'] + re.split(r'[\._]', d[6:])
        else:
            key = ['category']
        dict[' '.join(key)] = d
    return dict

def get_logic(type, pre_logic):
    type_pattern = get_type_pattern(type)
    logic = [[w] for w in type_pattern]
    if 'QUANTIFIER' in type_pattern:
        index = type_pattern.index('QUANTIFIER')
        logic[index] = [pre_logic[0]]
    elif type_pattern[0] == 'count':
        logic[0] = [pre_logic[0]]
    indexes = [i for i,w in enumerate(type_pattern) if w in ['ENTITY', 'VALUE', 'RELATION', 'TYPE']]
    for i in indexes:
        logic[i] = []
    cnt = 0
    for i, w in enumerate(pre_logic):
        if w not in ['?x', '?y', '|||'] and (i != 0 or w not in ['min', 'max', 'argmin', 'argmax', 'argless', 'argmore', 'count']):
            if (w.startswith('entity') or w.startswith('value')) and w not in ['entity', 'value']:
                w = w.upper()
                if len(logic[indexes[cnt]]) == 0:
                    try:
                        logic[indexes[cnt]].append(w)
                    except:
                        break
                        return 'False'
                    cnt += 1
                else:
                    try:
                        logic[indexes[cnt + 1]].append(w)
                    except:
                        break
                        return 'False'
                    cnt += 2
            else:
                try:
                    logic[indexes[cnt]].append(w)
                except:
                    break
                    return 'False'
        elif cnt < len(indexes) - 1 and len(logic[indexes[cnt]]) > 0:
            cnt += 1
    dict = get_relation()
    for i, w in enumerate(logic):
        if len(w) > 1:
            if ' '.join(w) in list(dict.keys()):
                logic[i] = [dict[' '.join(w)]]
            else:
                if w[0] != 'reverse':
                    logic[i] = ['mso:' + '_'.join(w)]
                else:
                    logic[i] = ['r-mso:' + '_'.join(w[1:])]
        elif len(w) == 1 and w[0] == 'category':
            logic[i] = [dict['category']]
    logic = ' '.join([ww for w in logic for ww in w])
    return logic


if __name__ == '__main__':
    with open('../DATA/GOLD/15_classes_dev_ns0.tsv', 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')
    for d in data:
        d = d.strip().split('\t')
        qp = d[1]
        pl = d[2].split(' ')
        ty = d[3]
        lgc = get_logic(ty, pl)
        with open('cache.txt', 'a', encoding='utf-8') as f:
            f.write(lgc + '\n')
