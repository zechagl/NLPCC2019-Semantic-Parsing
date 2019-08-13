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
sf = lambda x : '%.2f%%'%(x*100.)

""" ===== ===== ===== ===== ===== ===== """


def comb() :
    data_all = [[], [], []]
    # for file in tqdm(['5to10s_v2_ep9_0/dev_pred_v2.test_00.json', '5to10s_v2_ep9_0/dev_pred_v2.test_01.json', '5to10s_v2_ep9_0/dev_pred_v2.test_02.json', '5to10s_v2_ep9_0/dev_pred_v2.test_03.json', '5to10s_v2_ep9_0/dev_pred_v2.test_04.json', '5to10s_v2_ep9_0/dev_pred_v2.test_05.json', '5to10s_v2_ep9_0/dev_pred_v2.test_06.json', '5to10s_v2_ep9_1/dev_pred_v2.test_07.json', '5to10s_v2_ep9_1/dev_pred_v2.test_08.json', '5to10s_v2_ep9_1/dev_pred_v2.test_09.json', '5to10s_v2_ep9_1/dev_pred_v2.test_10.json', '5to10s_v2_ep9_1/dev_pred_v2.test_11.json', '5to10s_v2_ep9_1/dev_pred_v2.test_12.json', '5to10s_v2_ep9_1/dev_pred_v2.test_13.json', '5to10s_v2_ep9_2/dev_pred_v2.test_14.json', '5to10s_v2_ep9_2/dev_pred_v2.test_15.json', '5to10s_v2_ep9_2/dev_pred_v2.test_16.json', '5to10s_v2_ep9_0/dev_pred_v2.test_17.json', '5to10s_v2_ep9_1/dev_pred_v2.test_18.json', '5to10s_v2_ep9_3/dev_pred_v2.test_19.json']) :
    # for file in tqdm(['5to10s_v2_ep9_0/test_pred_v2.test_00.json', '5to10s_v2_ep9_0/test_pred_v2.test_01.json', '5to10s_v2_ep9_0/test_pred_v2.test_02.json', '5to10s_v2_ep9_0/test_pred_v2.test_03.json', '5to10s_v2_ep9_0/test_pred_v2.test_04.json', '5to10s_v2_ep9_0/test_pred_v2.test_05.json', '5to10s_v2_ep9_0/test_pred_v2.test_06.json', '5to10s_v2_ep9_1/test_pred_v2.test_07.json', '5to10s_v2_ep9_1/test_pred_v2.test_08.json', '5to10s_v2_ep9_1/test_pred_v2.test_09.json', '5to10s_v2_ep9_1/test_pred_v2.test_10.json', '5to10s_v2_ep9_1/test_pred_v2.test_11.json', '5to10s_v2_ep9_1/test_pred_v2.test_12.json', '5to10s_v2_ep9_1/test_pred_v2.test_13.json', '5to10s_v2_ep9_2/test_pred_v2.test_14.json', '5to10s_v2_ep9_2/test_pred_v2.test_15.json', '5to10s_v2_ep9_2/test_pred_v2.test_16.json', '5to10s_v2_ep9_2/test_pred_v2.test_17.json', '5to10s_v2_ep9_2/test_pred_v2.test_18.json', '5to10s_v2_ep9_2/test_pred_v2.test_19.json', '5to10s_v2_ep9_2/test_pred_v2.test_20.json']) :
    # for file in tqdm(['5to10s_v2_epc_0/dev_pred_v2.test_00.json', '5to10s_v2_epc_0/dev_pred_v2.test_01.json', '5to10s_v2_epc_0/dev_pred_v2.test_02.json', '5to10s_v2_epc_0/dev_pred_v2.test_03.json', '5to10s_v2_epc_0/dev_pred_v2.test_04.json', '5to10s_v2_epc_0/dev_pred_v2.test_05.json', '5to10s_v2_epc_0/dev_pred_v2.test_06.json', '5to10s_v2_epc_1/dev_pred_v2.test_07.json', '5to10s_v2_epc_1/dev_pred_v2.test_08.json', '5to10s_v2_epc_1/dev_pred_v2.test_09.json', '5to10s_v2_epc_1/dev_pred_v2.test_10.json', '5to10s_v2_epc_1/dev_pred_v2.test_11.json', '5to10s_v2_epc_1/dev_pred_v2.test_12.json', '5to10s_v2_epc_1/dev_pred_v2.test_13.json', '5to10s_v2_epc_2/dev_pred_v2.test_14.json', '5to10s_v2_epc_2/dev_pred_v2.test_15.json', '5to10s_v2_epc_2/dev_pred_v2.test_16.json', '5to10s_v2_epc_2/dev_pred_v2.test_17.json', '5to10s_v2_epc_2/dev_pred_v2.test_18.json', '5to10s_v2_epc_2/dev_pred_v2.test_19.json']) :
    # for file in tqdm(['5to10s_v2_epc_0/test_pred_v2.test_00.json', '5to10s_v2_epc_0/test_pred_v2.test_01.json', '5to10s_v2_epc_0/test_pred_v2.test_02.json', '5to10s_v2_epc_0/test_pred_v2.test_03.json', '5to10s_v2_epc_0/test_pred_v2.test_04.json', '5to10s_v2_epc_0/test_pred_v2.test_05.json', '5to10s_v2_epc_0/test_pred_v2.test_06.json', '5to10s_v2_epc_1/test_pred_v2.test_07.json', '5to10s_v2_epc_1/test_pred_v2.test_08.json', '5to10s_v2_epc_1/test_pred_v2.test_09.json', '5to10s_v2_epc_1/test_pred_v2.test_10.json', '5to10s_v2_epc_1/test_pred_v2.test_11.json', '5to10s_v2_epc_1/test_pred_v2.test_12.json', '5to10s_v2_epc_1/test_pred_v2.test_13.json', '5to10s_v2_epc_2/test_pred_v2.test_14.json', '5to10s_v2_epc_2/test_pred_v2.test_15.json', '5to10s_v2_epc_2/test_pred_v2.test_16.json', '5to10s_v2_epc_2/test_pred_v2.test_17.json', '5to10s_v2_epc_2/test_pred_v2.test_18.json', '5to10s_v2_epc_2/test_pred_v2.test_19.json', '5to10s_v2_epc_2/test_pred_v2.test_20.json']) :
    # for file in tqdm(['10to10s_v2_ep6_0/dev_pred_v2.test_00.json', '10to10s_v2_ep6_0/dev_pred_v2.test_01.json', '10to10s_v2_ep6_0/dev_pred_v2.test_02.json', '10to10s_v2_ep6_0/dev_pred_v2.test_03.json', '10to10s_v2_ep6_0/dev_pred_v2.test_04.json', '10to10s_v2_ep6_0/dev_pred_v2.test_05.json', '10to10s_v2_ep6_0/dev_pred_v2.test_06.json', '10to10s_v2_ep6_1/dev_pred_v2.test_07.json', '10to10s_v2_ep6_1/dev_pred_v2.test_08.json', '10to10s_v2_ep6_1/dev_pred_v2.test_09.json', '10to10s_v2_ep6_1/dev_pred_v2.test_10.json', '10to10s_v2_ep6_1/dev_pred_v2.test_11.json', '10to10s_v2_ep6_1/dev_pred_v2.test_12.json', '10to10s_v2_ep6_1/dev_pred_v2.test_13.json', '10to10s_v2_ep6_2/dev_pred_v2.test_14.json', '10to10s_v2_ep6_2/dev_pred_v2.test_15.json', '10to10s_v2_ep6_2/dev_pred_v2.test_16.json', '10to10s_v2_ep6_2/dev_pred_v2.test_17.json', '10to10s_v2_ep6_2/dev_pred_v2.test_18.json', '10to10s_v2_ep6_2/dev_pred_v2.test_19.json']) :
    for file in tqdm(['10to10s_v2_ep6_0/test_pred_v2.test_00.json', '10to10s_v2_ep6_0/test_pred_v2.test_01.json', '10to10s_v2_ep6_0/test_pred_v2.test_02.json', '10to10s_v2_ep6_0/test_pred_v2.test_03.json', '10to10s_v2_ep6_0/test_pred_v2.test_04.json', '10to10s_v2_ep6_0/test_pred_v2.test_05.json', '10to10s_v2_ep6_0/test_pred_v2.test_06.json', '10to10s_v2_ep6_1/test_pred_v2.test_07.json', '10to10s_v2_ep6_1/test_pred_v2.test_08.json', '10to10s_v2_ep6_1/test_pred_v2.test_09.json', '10to10s_v2_ep6_1/test_pred_v2.test_10.json', '10to10s_v2_ep6_1/test_pred_v2.test_11.json', '10to10s_v2_ep6_1/test_pred_v2.test_12.json', '10to10s_v2_ep6_1/test_pred_v2.test_13.json', '10to10s_v2_ep6_2/test_pred_v2.test_14.json', '10to10s_v2_ep6_2/test_pred_v2.test_15.json', '10to10s_v2_ep6_2/test_pred_v2.test_16.json', '10to10s_v2_ep6_2/test_pred_v2.test_17.json', '10to10s_v2_ep6_2/test_pred_v2.test_18.json', '10to10s_v2_ep6_2/test_pred_v2.test_19.json', '10to10s_v2_ep6_2/test_pred_v2.test_20.json']) :
        data = json_load(file)
        data_all[0] += data[0]
        data_all[1] += data[1]
        data_all[2] += data[2]

    print(list(map(len, data_all)))
    # json_dump(data_all, 'out.test.5_v2_ep9.dev_pred.json')
    # json_dump(data_all, 'out.test.5_v2_ep9.test_pred.json')
    # json_dump(data_all, 'out.test.5_v2_epc.dev_pred.json')
    # json_dump(data_all, 'out.test.5_v2_epc.test_pred.json')
    # json_dump(data_all, 'out.test.10_v2_ep6.dev_pred.json')
    json_dump(data_all, 'out.test.10_v2_ep6.test_pred.json')


def recover(comb_file, raw_file, len_file, out_path) :
    raw_data = codecs_in(raw_file).readlines()
    raw_data = [t.strip().split('\t') for t in raw_data]

    scores = json_load(comb_file)[0]

    for raw_l, pred in tqdm(zip(raw_data, scores)) :
        raw_l.append(pred[-1])
    
    """"""

    temp = 0
    def get_next(n, temp) :
        to_ret = raw_data[temp:temp+n]
        temp = temp+n
        return to_ret, temp

    len_info = json_load(len_file)
    results = []
    for dt in tqdm(len_info) :
        result_temp = []

        for len_t in dt['lens'] :
            datas, temp = get_next(len_t, temp)
            datas = sorted(datas, key=lambda x:x[-1], reverse=True)
            datas = [t for i, t in enumerate(datas) if any([i < 10, t[-1]>0.04])]
            #datas = [t for i, t in enumerate(datas)]
            result_temp.append({
                    'qu'   : datas[0][1],
                    'type' : datas[0][-2],
                    'pred' : [[t[2], t[-1]] for t in datas],
                })

        results.append(result_temp)
    """"""

    json_dump(results, out_path)

def eval(pred_result, raw_ts2) :
    results = json_load(pred_result)
    datas   = json_load(raw_ts2)
    recover_dict = json_load('dictt.json')

    dict_counts = {}
    counts     = {t:0 for t in ['all', 'can', 'top1', 'top2', 'top3', 'top5', 'top10', '#cand.5', 'rec.5']}
    for pred_t, dt in zip(results, datas) :
        temp_counts = {t:0 for t in ['all', 'can', 'top1', 'top2', 'top3', 'top5', 'top10', '#cand.5', 'rec.5']}

        temp_counts['all'] += 1
        if ' '.join(dt['question_pattern']) == ' '.join(dt['question_pre_pattern']) :
            temp_counts['can'] += 1

        for i, t in enumerate(pred_t[0]['pred']) :
            recover = recover_dict[pred_t[0]['type']][t[0]]
            if i == 0 or t[1] > 0.5 :
                temp_counts['#cand.5'] += 1
            
            if recover == ' '.join(dt['logical_pattern']) :
                if i < 1 :
                    temp_counts['top1'] += 1
                if i < 2 :
                    temp_counts['top2'] += 1
                if i < 3 :
                    temp_counts['top3'] += 1
                if i < 5 :
                    temp_counts['top5'] += 1
                if i < 10 :
                    temp_counts['top10'] += 1
                if i == 0 or t[1] > 0.5 :
                    temp_counts['rec.5'] += 1


        for k in temp_counts :
            counts[k] += temp_counts[k]
        if not dt['type'] in dict_counts :
            dict_counts[dt['type']] = temp_counts
        else :
            for k in temp_counts :
                dict_counts[dt['type']][k] += temp_counts[k]

    def show(counts) :
        show_rate = lambda x, rate=100. : ' \t%d \t%.2f'%(x, x*rate/counts['all'])

        for k in ['can', 'top1', 'top2', 'top3', 'top5', 'top10'] :
            print(k, show_rate(counts[k]))
        
        print('rec.5', show_rate(counts['rec.5']))
        print('#cand.5', show_rate(counts['#cand.5'], 1))


        for k in ['top1'] :
            print(k, show_rate(counts[k]))


    show(counts)
    # print('\n')
    for k in sorted(dict_counts.keys()) :
    #     print('\n\n\n')
        print(k)
        show(dict_counts[k])

    return results


def combp(input_files, output_files) :
    files = [json_load(file_t) for file_t in input_files]

    new_results = []
    for i in range(len(files[0])) :
        result_t = []
        for j in range(len(files[0][i])) :
            dts = [t[i][j] for t in files]
            temp = {
                'pred' : {},
                'qu'   : dts[0]['qu'],
                'type' : dts[0]['type'],
            }
            preds = [pred for dt in dts for pred in dt['pred']]

            for k, s in preds :
                if not k in temp['pred'] :
                    temp['pred'][k] = s
                
                # temp['pred'][k] = (s * temp['pred'][k]) ** 0.5
                temp['pred'][k] = (s + temp['pred'][k]) * 0.5
            
            temp['pred'] = sorted([[k, temp['pred'][k]] for k in temp['pred']], key=lambda x:x[1], reverse=True)
            result_t.append(temp)
        new_results.append(result_t)
    json_dump(new_results, output_files)




def combp2(input_files, output_files) :
    files = [json_load(file_t) for file_t in input_files]

    combt = lambda x : (x[0]+x[1]+x[2]) / 3

    new_results = []
    for i in range(len(files[0])) :
        result_t = []
        for j in range(len(files[0][i])) :
            dts = [t[i][j] for t in files]
            temp = {
                'pred' : {},
                'qu'   : dts[0]['qu'],
                'type' : dts[0]['type'],
            }

            for ii, pred_i in enumerate(dts) :
                for k, s in pred_i['pred'] :
                    if not k in temp['pred'] :
                        temp['pred'][k] = []
                    
                    # temp['pred'][k] = (s * temp['pred'][k]) ** 0.5
                    while not len(temp['pred'][k]) == ii :
                        temp['pred'][k].append(0)
                    temp['pred'][k].append(s)
            for k in temp['pred'] :
                while not len(temp['pred'][k]) == 3 :
                    temp['pred'][k].append(0)

            temp['pred'] = {k:combt(temp['pred'][k]) for k in temp['pred']}
            temp['pred'] = sorted([[k, temp['pred'][k]] for k in temp['pred']], key=lambda x:x[1], reverse=True)
            temp['pred'] = [t for i, t in enumerate(temp['pred']) if any([t[1]>0.01, i < 3])]
            result_t.append(temp)
        new_results.append(result_t)
    json_dump(new_results, output_files)


if __name__ == '__main__':
    recover('../../data/json/pattern_pair_merge_5_dev.json', '../../data/patternpair/dev.tsv',
            '../../data/json/dev_length.json', '../../data/json/pattern_pair_merge_5_dev_t3_pre.json')
    # recover('../../data/json/pattern_pair_merge_5_test.json', '../../data/patternpair/test.tsv',
    #         '../../data/json/test_length.json', '../../data/json/pattern_pair_merge_5_test_t3_pre.json')
