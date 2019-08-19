#coding=utf-8
#-*- coding: UTF-8 -*-
import sys

"""python3"""

import re
import os
import json
import time
import codecs
import random
import argparse
import numpy as np
from tqdm import tqdm
from tqdm import trange
from collections import Counter



codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)

char_num=set('abcdefghjiklmnopqrstuvwxyz1234567890.')


""" ===== ===== ===== ===== ===== ===== """

def get_pattern(qu, logic, param) :
    qu_copy    = [t for t in qu]
    logic_copy = [t for t in logic]

    q2_start   = qu_copy.index('|||') + 1 if '|||' in qu_copy else -1
    entity_set = set()
    param = sorted(param, key=lambda t:t['range'][0]+100 if t['q'] == '@Q2' else t['range'][0])
    for t in param :
        if t['range'][0] == -1 :
            assert not t['type'] == 'entity'
            continue
        if t['type'] == 'type' :
            continue
        if t['text'] in entity_set :
            continue
        entity_set.add(t['text'])

        bias = q2_start if t['q'] == '@Q2' else 0
        i_start = bias+t['range'][0]
        i_end   = bias+t['range'][1]+1
        infor = qu_copy[i_start:i_end]

        assert '_'.join(infor) == t['text']
        assert t['type'] in ['entity', 'value']

        qu_copy[i_start] = t['type'].upper()+str(len(entity_set))
        for i in range(i_start+1, i_end) :
            qu_copy[i] = 'TO_REMOVE'

        for i, tt in enumerate(logic_copy) :
            if tt == t['text'] :
                logic_copy[i] = t['type'].upper()+str(len(entity_set))
    qu_copy = [t for t in qu_copy if not t == 'TO_REMOVE']
    return qu_copy, logic_copy


'''f1'''
def get_type_pattern(type_in) :
    assert type_in in ['single-relation', 'superlative0', 'superlative1', 'superlative2', 'superlative3', 'multi-hop', 'aggregation', 'cvt', 'yesno', 'multi-constraint', 'multi-choice', 'comparative', 'multi-turn-entity', 'multi-turn-answer', 'multi-turn-predicate']

    if type_in == 'single-relation' :
        return ['(', 'lambda', '?x', '(', 'RELATION', 'ENTITY', '?x', ')', ')']
    elif type_in == 'superlative0' :
        return ['(', 'QUANTIFIER', '(', 'lambda', '?x', '(', 'RELATION', '?x', 'ENTITY', ')', ')', '(', 'lambda', '?x', '(', 'lambda', '?y', '(', 'RELATION', '?x', '?y', ')', ')', ')', 'VALUE', ')']
    elif type_in == 'superlative1' :
        return ['(', 'QUANTIFIER', '(', 'lambda', '?x', '(', 'isa', '?x', 'RELATION', ')', ')', '(', 'lambda', '?x', 'lambda', '?y', '(', 'RELATION', '?x', '?y', ')', ')', 'VALUE', ')']
    elif type_in == 'superlative2' :
        return ['(', 'QUANTIFIER', '(', 'lambda', '?x', '(', 'isa', '?x', 'RELATION', ')', ')', '(', 'lambda', '?x', '(', 'lambda', '?y', '(', 'RELATION', '?x', '?y', ')', ')', ')', 'VALUE', ')']
    elif type_in == 'superlative3' :
        return ['(', 'QUANTIFIER', '(', 'lambda', '?x', '(', 'and', '(', 'RELATION', '?x', 'ENTITY', ')', '(', 'isa', '?x', 'RELATION', ')', ')', ')', '(', 'lambda', '?x', '(', 'lambda', '?y', '(', 'RELATION', '?x', '?y', ')', ')', ')', 'VALUE', ')']
    elif type_in == 'multi-hop' :
        return ['(', 'lambda', '?x', 'exist', '?y', '(', 'and', '(', 'RELATION', 'ENTITY', '?y', ')', '(', 'RELATION', '?y', '?x', ')', ')', ')']
    elif type_in == 'aggregation' :
        return ['count', '(', 'lambda', '?x', '(', 'RELATION', 'ENTITY', '?x', ')', ')']
    elif type_in == 'cvt' :
        return ['(', 'lambda', '?x', 'exist', '?y', '(', 'and', '(', 'RELATION', 'ENTITY', '?y', ')', '(', 'RELATION', '?y', 'ENTITY', ')', '(', 'RELATION', '?y', '?x', ')', ')', ')']
    elif type_in == 'yesno' :
        return ['(', 'RELATION', 'ENTITY', 'ENTITY', ')']
    elif type_in == 'multi-constraint' :
        return ['(', 'lambda', '?x', '(', 'and', '(', 'RELATION', 'ENTITY', '?x', ')', '(', 'RELATION', 'ENTITY', '?x', ')', ')', ')']
    elif type_in == 'multi-choice' :
        return ['(', 'lambda', '?x', '(', 'and', '(', 'RELATION', 'ENTITY', '?x', ')', '(', 'or', '(', 'equal', '?x', 'ENTITY', ')', '(', 'equal', '?x', 'ENTITY', ')', ')', ')', ')']
    elif type_in == 'comparative' :
        return ['(', 'QUANTIFIER', '(', 'lambda', '?x', '(', 'isa', '?x', 'TYPE', ')', ')', '(', 'lambda', '?x', 'lambda', '?y', '(', 'RELATION', '?x', '?y', ')', ')', 'VALUE', ')']
    elif type_in == 'multi-turn-entity' :
        return ['(', 'lambda', '?x', '(', 'RELATION', 'ENTITY', '?x', ')', ')', '|||', '(', 'lambda', '?x', '(', 'RELATION', 'ENTITY', '?x', ')', ')']
    elif type_in == 'multi-turn-answer' :
        return ['(', 'lambda', '?x', '(', 'RELATION', 'ENTITY', '?x', ')', ')', '|||', '(', 'lambda', '?x', 'exist', '?y', '(', 'and', '(', 'RELATION', 'ENTITY', '?y', ')', '(', 'RELATION', '?y', '?x', ')', ')', ')']
    elif type_in == 'multi-turn-predicate' :
        return ['(', 'lambda', '?x', '(', 'RELATION', 'ENTITY', '?x', ')', '|||', '(', 'lambda', '?x', '(', 'RELATION', 'ENTITY', '?x', ')', ')']
    else :
        assert False


'''f2'''
def check_pattern(pattern, logic, param) :
    if not len(pattern) == len(logic) :
        return False

    param_dict = {t['text']:t['type'].upper() for t in param}

    check_info = []
    for pt, lt in zip(pattern, logic) :
        if pt == lt :
            check_info.append(True)
            # 全等，则认为相等
        elif pt == 'RELATION' and any([lt.startswith('mso:'), lt.startswith('r-mso:'), lt.startswith('dev:')]) :
            # 并列第二优先级： 关系上的匹配
            check_info.append(True)
        elif pt == 'QUANTIFIER' and lt in ['argmore', 'argless', 'argmin', 'argmax', 'max', 'min']:
            # 并列第二优先级： 量词的匹配
            check_info.append(True)
        elif not lt in param_dict :
            # 第三优先级：不算是关系，且不再param当中，就认为是错的
            check_info.append(False)
        elif param_dict[lt]==pt :
            # 第四优先级：如果type相同，认为是对的
            check_info.append(True)
        elif param_dict[lt]=='VALUE' and pt == 'ENTITY':
            # 第五优先级：Entity的槽位是可以由value来填充
            check_info.append(True)
        else :
            check_info.append(False)

    return all(check_info)


'''f3'''
# def get_te_type_pattern(entity_pred, type_pred_pattern, type, question, id):
#     te_pattern = [w for w in type_pred_pattern]
#     en_index = [i for i in range(len(type_pred_pattern)) if type_pred_pattern[i] == 'ENTITY']
#     gold_num = len(en_index)
#     index = choose_entity(entity_pred, type, question, gold_num, id)
#     for i in range(gold_num):
#         if index[i] >= 0:
#             te_pattern[en_index[i]] = entity_pred[index[i]]['text']
#     return te_pattern


def deal_file_ts1(input_path, output_path) :
    data = json_load(input_path)

    for dt in tqdm(data) :
        dt['question_pattern'], dt['logical_pattern'] = get_pattern(dt['question'], dt['logical'], dt['parameters'])

        dt['type_pattern'] = get_type_pattern(dt['type'])

        dt['type_pattern_correct'] = check_pattern(dt['type_pattern'], dt['logical'], dt['parameters'])

    print('rate of type_pattern_correct: ', len([1 for dt in data if dt['type_pattern_correct']]) / float(len(data)))
    json_dump(data, output_path)

    return data




if __name__ == '__main__':
    deal_file_ts1('../data/time_step/15_classes_dev_timestep0.json', '../data/time_step/15_classes_dev_timestep1.json')
    deal_file_ts1('../data/time_step/15_classes_train_timestep0.json', '../data/time_step/15_classes_train_timestep1.json')
