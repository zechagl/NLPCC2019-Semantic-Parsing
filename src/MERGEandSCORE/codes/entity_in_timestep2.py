import json
import codecs
import re
import argparse


json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
list = ['first', 'last', 'most', 'top', 'recent', '1st', 'new', 'recently', 'second']

# padding, b, m, o
def get_prob(prob, ref):
    output = []
    for i in range(9000):
        prb = []
        ref[i] = ref[i][1:]
        prob[i] = prob[i][1:]
        for j, rf in enumerate(ref[i]):
            if rf != 0:
                pb = prob[i][j]
                if rf != 4:
                    prb.append([pb])
                else:
                    prb[-1].append(pb)
        output.append(prb)
    for i, prob in enumerate(output):
        for j, prb in enumerate(prob):
            num = len(prb)
            prb = [[p[k] for p in prb] for k in range(4)]
            for k, p in enumerate(prb):
                sum = 0
                for l in range(num):
                    sum += p[l]
                avg = sum / num
                prb[k] = avg
            output[i][j] = prb
    return output

def entity_match(data):
    x, y, z = 0, 0, 0
    for sample in data:
        ptypes = [t['type'] for t in sample['parameters']]
        qu = sample['question']
        dil = len(qu)
        if '|||' in qu:
            dil = qu.index('|||')

        question_pattern = sample['question_pattern']
        entity_pred = sample['entity_pred']
        question_pattern_pred = [w for w in qu]

        for prd in entity_pred:
            r = prd['range']
            b, e = r[0], r[1]
            if prd['q'] == '@Q2':
                b += dil + 1
                e += dil + 1
            elif prd['text'].count('|||') > 0:
                e += dil + 1
            question_pattern_pred[b] = '<this_is_B>'
            for i in range(b + 1, e + 1):
                question_pattern_pred[i] = '<this_is_I>'

        prediction = []
        cnt = 1
        for w in question_pattern_pred:
            if w == '<this_is_B>':
                prediction.append('ENTITY' + str(cnt))
                cnt += 1
            elif w != '<this_is_I>':
                prediction.append(w)
        pred_cnt = cnt - 1

        prediction = ' '.join(prediction)

        gold = []
        cnt = 1
        for w in question_pattern:
            if w.startswith('VALUE') or w.startswith('ENTITY'):
                gold.append('ENTITY' + str(cnt))
                cnt += 1
            else:
                gold.append(w)
        gold = ' '.join(gold)


        if cnt != 0:
            if gold == prediction or 'type' in ptypes:
                x += 1
            else:
                if pred_cnt != cnt - 1:
                    z += 1
                    gold = ' '.join(question_pattern)
                    with open('data/cache/error_removetype_v2_10_5.10.10_timestep2.txt', 'a', encoding='utf-8') as f:
                        f.write('<gold>\t' + gold + '\n' + '<prediction>\t' + prediction + '\n')
                        f.write('===================================================\n')
            y += 1
    print(x, y, 1-x/y, z/y)

def value_filter_for_superlative(data):
    output = []
    for sample in data:
        qu = sample['question']
        entity_pred = sample['entity_pred']
        types = [prd['type'] for prd in entity_pred]
        if 'value' in types:
            flag = True
            for word in qu:
                if (word in list) or word.endswith('est') or word.count('min')+word.count('max') > 0:
                    flag = False
                    break
            if flag:
                sample['entity_pred'] = [prd for prd in entity_pred if prd['type'] != 'value']
        output.append(sample)
    return output

def entity_num(data, prob, mode):
    output = []
    for i, sample in enumerate(data):
        question = sample['question']
        entity_pred = sample['entity_pred']

        label = ['o' for w in question]
        for j, ep in enumerate(entity_pred):
            if re.match(r'^[0-9\./_-]+$', ep['text']):
                entity_pred[j]['type'] = 'value'
            for k in range(ep['range'][0], ep['range'][1] + 1):
                index = k
                if ep['q'] == '@Q2':
                    index = k + question.index('|||') + 1
                if k == ep['range'][0]:
                    label[index] = 'b'
                elif k == ep['range'][1]:
                    label[index] = 'e'
                else:
                    label[index] = 'i'

        type_pred = sample['type_pred']
        type_pred_pattern = sample['type_pred_pattern']
        entitynum = type_pred_pattern.count('ENTITY')
        if sample['type_pred'] in ['multi-turn-answer', 'multi-turn-entity']:
            entitynum -= 1
        valuenum = type_pred_pattern.count('VALUE')

        values = [ep for ep in entity_pred if ep['type'] == 'value']
        entities = [ep for ep in entity_pred if ep['type'] == 'entity']
        # align the number of entities for superlative
        if type_pred.startswith('superlative'):
            search = [v for v in values if re.match(r'^[0-9]+$', v['text'])]
            if len(values) == 0 or len(search) == 0:
                flag = False
                for w in question:
                    if w in list or w.count('min')+w.count('max') > 0 or w.endswith('est'):
                        flag = True
                        break
                if flag:
                    if 'second' in question:
                        values += [{'text':'2', 'type':'value', 'range':[-1,-1], 'q':None}]
                        index = question.index('second')
                        label[index] = 'b'
                    else:
                        values += [{'text':'1', 'type':'value', 'range':[-1,-1], 'q':None}]
                else:
                    numbers = [int(w) for j,w in enumerate(question) if re.match(r'^[0-9]+$', w) and label[j] == 'o' and j < len(question) - 1 and question[j + 1] in ['th', 'nd', 'st']]
                    min_n = 100000
                    if len(numbers) > 0:
                        for n in numbers:
                            if n < min_n:
                                min_n = n
                        index = question.index(str(min_n))
                        value.append({'text':str(min_n), 'type':'value', 'range':[index,index], 'q':None})
                    elif len(entities) > entitynum:
                        for j,e in enumerate(entities):
                            rg = e['range']
                            txt = e['text'].split('_')
                            flag = False
                            for k in range(rg[0], rg[1] + 1):
                                if re.match(r'^[0-9]+$', question[k]) and k < len(question) - 1 and question[k + 1] in ['th', 'nd', 'st']:
                                    flag = True
                                    label[k] = 'b'
                                    break
                            if flag:
                                values.append({'text':question[k], 'type':'value', 'range':[k,k], 'q':None})
                                if k == rg[0]:
                                    rg[0] += 1
                                else:
                                    for l in range(k+1, rg[1]+1):
                                        label[l] = 'o'
                                    rg[1] = k - 1
                                entities[j]['text'] = '_'.join(question[rg[0] : rg[1] + 1])
                                entities[j]['range'] = [rg[0], rg[1]]
                                break
                        if len(values) == 0:
                            values += [{'text':'1', 'type':'value', 'range':[-1,-1], 'q':None}]
                    else:
                        values.append({'text':'1', 'type':'value', 'range':[-1,-1], 'q':None})
        # align the number of entities for comparative
        elif type_pred == 'comparative':
            if len(values) == 0:
                numbers = [w for j,w in enumerate(questions) if re.match(r'^[0-9\./_-]+$', w)]
                for j, e in enumerate(entities):
                    rg = e['range']
                    k_list = []
                    for k in range(rg[0], rg[1] + 1):
                        if question[k] in numbers:
                            k_list.append(k)
                            label[k] = 'b'
                    if len(k_list) > 0:
                        if rg[0] == k_list[0] and rg[1] == k_list[1]:
                            entities[j]['type'] = '!entity'
                        elif rg[0] != k_list[0]:
                            for l in range(k_list[-1]+1, rg[1]+1):
                                label[l] = 'o'
                            rg[1] = k_list[0] - 1
                            entities[j]['range'] = [rg[0], rg[1]]
                            entities[j]['text'] = '_'.join(question[rg[0]:rg[1]+1])
                        else:
                            for l in range(rg[0], k_list[0]):
                                label[l] = 'o'
                            rg[0] = k_list[-1] + 1
                            entities[j]['range'] = [rg[0], rg[1]]
                            entities[j]['text'] = '_'.join(question[rg[0]:rg[1]+1])

                index1 = question.index(numbers[0])
                index2 = quetsion.index(numbers[-1])
                values.append({'text':'_'.join(numbers), 'type':'value', 'range':[index1,index2], 'q':None})

        entities = [e for e in entities if e['type'] != '!entity']

        final = []
        # when entities are more than need
        if entitynum + valuenum < len(entities + values):
            final = []
            prb = []
            minus_one = 0
            for j, en in enumerate(entities + values):
                rg = en['range']
                i1, i2 = rg[0], rg[1]
                if en['q'] == '@Q2':
                    i1 += question.index('|||') + 1
                    i2 += question.index('|||') + 1
                if rg[0] != -1:
                    prb.append((j, prob[i][i1][1]))
                else:
                    minus_one += 1
                    final.append(en)
                    prb.append((j, 0))
            prb.sort(key=lambda x:x[1], reverse=True)
            x = entities + values
            for j in range(entitynum + valuenum - minus_one):
                final.append(x[prb[j][0]])
            sample['entity_pred'] = final
            if len(final) != entitynum + valuenum:
                print(final)
        # when entities are less than need
        elif entitynum + valuenum > len(entities + values):
            final = entities + values
            prb = []
            for j, w in enumerate(question):
                if label[j] == 'o' and question[j] != '|||':
                    prb.append((j, prob[i][j][1]))

            prb.sort(key=lambda x:x[1], reverse=True)
            for j in range(entitynum + valuenum - len(entities + values)):
                r1 = prb[j][0]
                r2 = r1
                txt = question[r1]
                qq = None
                if '|||' in question:
                    qq = '@Q1'
                    if r1 > question.index('|||'):
                        r1 -= question.index('|||') + 1
                        r2 -= question.index('|||') + 1
                        qq = '@Q1'
                ty = 'entity'
                if re.match(r'^[0-9\./_-]+$', txt):
                    ty = 'value'
                final.append({'text':txt, 'type':ty, 'range':[r1,r2], 'q':qq})
            sample['entity_pred'] = final
        else:
            sample['entity_pred'] = entities + values
        question_pre_pattern = []
        label = ['o' for w in question]
        for en in sample['entity_pred']:
            rg = en['range']
            i1, i2 = rg[0], rg[1]
            if i1 != -1:
                if en['q'] == '@Q2':
                    i1 += question.index('|||') + 1
                    i2 += question.index('|||') + 1
                for j in range(i1, i2 + 1):
                    if j == i1:
                        if en['type'] == 'entity':
                            label[j] = 'be'
                        else:
                            label[j] = 'bv'
                    else:
                        label[j] = 'i'
        cnt = 1
        for j, w in enumerate(question):
            if label[j] == 'o':
                question_pre_pattern.append(w)
            elif label[j] == 'be':
                question_pre_pattern.append('ENTITY' + str(cnt))
                cnt += 1
            elif label[j] == 'bv':
                question_pre_pattern.append('VALUE' + str(cnt))
                cnt += 1
        sample['question_pre_pattern'] = question_pre_pattern
        output.append(sample)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='entity_in_timestep2.py')
    parser.add_argument('-mode', default='dev', help="""dev or test""")
    parser.add_argument('-input_path', default='', help="""input timestep2-v2 in json""")
    parser.add_argument('-label_path', default='', help="""input label result in json""")
    parser.add_argument('-result_path', default='', help="""output timestep2-v3 in json""")
    opt = parser.parse_args()

    labels = json_load(opt.label_path)
    # get probability
    prob = labels[2]
    ref = labels[3]
    prob = get_prob(prob, ref)
    data = json_load(opt.input_path)
    data = entity_num(data, prob, opt.mode)
    json_dump(data, opt.result_path)
