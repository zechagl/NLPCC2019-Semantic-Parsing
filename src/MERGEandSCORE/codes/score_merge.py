import os
import time
import json
import codecs
import sys
import math
import re
import argparse
from tqdm import tqdm
import logging



json_load = lambda x:json.load(codecs.open(x, 'r', encoding='utf-8'))


def loss_score(prediction):
    '''根据seq2seq模型的loss数据计算loss-score'''
    pred = prediction[-1]
    norm = 8    # 具体需根据loss数值来定 --> normalize
    s = 1 -((pred[0] + pred[1] + pred[2]) / 3) / norm
    return s

def get_cover_prob(question_words, logical):
    '''计数预测结果对源question中的词的覆盖数'''
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

def return_to_raw(logical, type, id, mode):
    '''将之前转换（预处理）过的multi-turn-predicate和multi-choice这两个类的logical form形式复原'''
    if type == 'multi-turn-predicate':
        assert '|||' in logical
        pos = logical.index('|||')
        logical = logical[:pos] + [')'] + logical[pos:]
    if mode == 'dev':
        if type == 'multi-choice' and id in [814, 1591, 2108, 2265, 2273, 3048, 6186, 6410, 6458, 6608, 6681, 6780, 7511, 7541, 7543, 7655, 7709, 7884]:
            logical[7], logical[8] = logical[8], logical[7]
            if logical[6].startswith('mso'):
                logical[6] = 'r-' + logical[6]
            elif logical[6].startswith('r-mso'):
                logical[6] = logical[6][2:]
    return logical

def filter_error(sample):
    '''计算三种错误率（ f3属于(f2并f1) ）
    f1：type预测错误
    f2：entity预测错误
    f3：logical form预测错误
    '''
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
    '''根据四种分数：todo1，todo3，loss和覆盖词数计算总分数'''
    ##=== score for todo3 - pattern score ===##
    s1 = -100 + s1 if s1 < 0.5 else s1
    if opt.remove_pair and opt.remove_pointer:
        return s1, 0
    ##=== score for todo1 - entity-predicate pair score ===##
    tmp_s2 = 1
    for x in s2:
        tmp_s2 *= x
    cnt2 = len(s2)
    s2 = tmp_s2
    s2 = math.pow(s2, 1 / cnt2) if cnt2 != 0 else 1
    
    para1 = opt.todo1to3
    if type == 'aggregation':
        para1 = opt.todo1to3_aggregation
    elif type == 'single-relation':
        para1 = opt.todo1to3_singlerelation
    elif type == 'superlative0':
        para1 = opt.todo1to3_superlative0
    elif type == 'multi-choice':
        para1 = opt.todo1to3_multichoice
    s = s1 + s2 * para1
    if opt.remove_pointer:
        return s, tmp_s2
    ##=== score for cover rate ===##
    para2 = opt.cover2todo13
    s = s/(1 + para1) + para2 * s3
    ##=== score for loss - pointer score ===##
    para3 = opt.loss2others
    if opt.remove_pair:
        if type == 'aggregation':
            para3 = 1.5
        else:
            para3 = 15
        s = s1 + s4 * para3
        return s, tmp_s2
    s = s/(1 + para2) + s4 * para3

    return s, tmp_s2

def process(sample, opt):
    '''
    计算分数选出分数最高者作为最终预测结果
    分为dev和test两种模式处理，dev因需要得到错误率和错误原因，故需更多计算
    '''
    if opt.mode == 'dev':
        flag = filter_error(sample)

    type_pred = sample['type_pred']
    predlist = sample['logical_pred']
    ### ===== 将mult-turn-entity这种需要预测两种句子的两组结果（todo3的结果）进行排列组合 ===== ###
    if type_pred in ['multi-turn-entity'] and 'logical_pred_1' in sample and 'logical_pred_0' in sample:
        q1list = sample['logical_pred_0']
        q2list = sample['logical_pred_1']
        loss_for_split = sample['loss_for_split']
        cnt = 0
        for q1 in q1list:
            for q2 in q2list:
                LP = q1[0] + ' ||| ' + q2[0]
                S = q1[1] * q2[1]
                ER = q1[2] + q2[2]
                LS = loss_for_split[cnt]
                cnt += 1
                predlist.append([LP, S, ER, LS])
        predlist_dict={}
        for lp,s,er,ls in predlist:
            if not lp in predlist_dict:
                predlist_dict[lp] = [0, er, ls]
            predlist_dict[lp][0] += s
        predlist = [[k, predlist_dict[k][0], predlist_dict[k][1], predlist_dict[k][2]] for k in predlist_dict]
    ### ===== 计算总分数 ===== ###
    target, max_score = -1, -100000000000
    for index, pred in enumerate(predlist):
        s1 = pred[1]
        s2 = [er[3] for er in pred[2]]
        s3 = get_cover_prob(sample['question_pre_pattern'], pred[0])
        s4 = loss_score(pred)
        final_score, s2 = score(s1, s2, s3, s4, type_pred, opt)

        predlist[index] += [s2, s3, s4]
        if final_score > max_score:
            max_score = final_score
            target = index
    logical_final_pred = predlist[target][0]
    
    if opt.mode == 'dev':
        ### ===== 统计预测情况（针对dev） ===== ###
        ### ===== 四种分数的排名 ===== ###
        if all(flag):
            s1_list = [pred[1] for pred in predlist]
            s2_list = [pred[4] for pred in predlist]
            s3_list = [pred[5] for pred in predlist]
            s4_list = [pred[6] for pred in predlist]
            lp_list = [pred[0] for pred in predlist]
            pos = lp_list.index(' '.join(sample['logical']))
            s1, s2, s3, s4 = s1_list[pos], s2_list[pos], s3_list[pos], s4_list[pos]
            s1_list.sort(key=lambda x:-x)
            s2_list.sort(key=lambda x:-x)
            s3_list.sort(key=lambda x:-x)
            s4_list.sort(key=lambda x:-x)
            rk1, rk2, rk3, rk4 = s1_list.index(s1) + 1, s2_list.index(s2) + 1, s3_list.index(s3) + 1, s4_list.index(s4) + 1
            h1, h2, h3, h4 = s1_list[0], s2_list[0], s3_list[0], s4_list[0]
        else:
            rk1, rk2, rk3, rk4 = -1, -1, -1, -1
            s1, s2, s3, s4 = 0, 0, 0, 0
            h1, h2, h3, h4 = 0, 0, 0, 0
        return logical_final_pred, flag, (rk1, rk2, rk3, rk4), (s1, s2, s3, s4), (h1, h2, h3, h4)
    else:
        return logical_final_pred


if __name__ == '__main__':
    ### ===== Prepare Options ===== ###
    parser = argparse.ArgumentParser(description='score_merge.py')
    parser.add_argument('-mode', required=True, choices=['dev', 'test'])

    parser.add_argument('-input_path', default='', 
                        help="""input timestep2-merge in json""")
    parser.add_argument('-result_path', default='', 
                        help="""output in txt""")
    parser.add_argument('-error_path', default='', 
                        help="""error samples in txt""")
    parser.add_argument('-eval_path', default='', 
                        help="""evaluation results on development set""")

    parser.add_argument('-todo1to3', type=float, default=0.4,
                        help="""TODO1 : TODO3""")
    parser.add_argument('-todo1to3_aggregation', type=float, default=0.5,
                        help="""TODO1 : TODO3 for aggregation""")
    parser.add_argument('-todo1to3_singlerelation', type=float, default=2,
                        help="""TODO1 : TODO3 for single-relation""")
    parser.add_argument('-todo1to3_superlative0', type=float, default=15,
                        help="""TODO1 : TODO3 for superlative0""")
    parser.add_argument('-todo1to3_multichoice', type=float, default=0.25,
                        help="""TODO1 : TODO3 for multi-choice""")
    parser.add_argument('-cover2todo13', type=float, default=0.01,
                        help="""cover-score : (TODO1 & TODO3)""")
    parser.add_argument('-loss2others', type=float, default=1.5,
                        help="""loss-score : [cover-score & (TODO1 & TODO3)]""")
    
    parser.add_argument('-remove_pointer', action='store_true', 
                        help="""TEST POINTER - remove loss-score from the final score""")
    parser.add_argument('-remove_pair', action='store_true', 
                        help="""TEST EP-PAIR - remove entity-predicate pair-score from the final score""")
    
    opt = parser.parse_args()

    ### ===== load待预测数据，准备相关变量用于记录结果 ===== ###
    samples = json_load(opt.input_path)
    result_filename, error_filename = opt.result_path, opt.error_path
    x, y = 0, 0
    dict_acc = {}
    dict_acc['superlative'] = [0, 0]
    not_in_cnt, not_type_cnt, not_entity_cnt, highest_loss = 0, 0, 0, 0
    
    for index, sample in tqdm(enumerate(samples), desc='   - (Predicting) -   '):
        type = sample['type']
        id = sample['id']
        logical = sample['logical_raw']
        ### ===== 预测 ===== ###
        if opt.mode == 'dev':
            logical_pred, flag, rank, scores, highest = process(sample, opt)
        else:
            logical_pred = process(sample, opt)
        ### ===== 统计错误情况（dev） ===== ###
        if opt.mode == 'dev':
            if not flag[2]:
                not_in_cnt += 1
            if not flag[0]:
                not_type_cnt += 1
            if not flag[1]:
                not_entity_cnt += 1            
        ### ===== 后处理还原 ===== ###
        logical_pred = logical_pred.split(' ')
        logical_pred = return_to_raw(logical_pred, type, id, opt.mode)
        logical_pred = ' '.join(logical_pred)        
        ### ===== 生成最终结果 ===== ###
        entity_pred = sample['entity_pred']
        print_entity_pred = []
        for i, en in enumerate(entity_pred):
            ### ===== 将entity转换为相应的单词 ===== ###
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
            ### ===== 计算entity span（考虑-1等特殊情况） ===== ###
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
            ### ===== 得到entity span ===== ###
            txt = [pep[0], pep[1]]
            rg = pep[2]
            txt.append('[' + str(rg[0]) + ',' + str(rg[1]) + ']')
            if len(pep) > 3:
                txt.append(pep[3])
            txt = ' '.join(txt)
            print_entity_pred[i] = txt
        print_entity_pred = ' ||| '.join(print_entity_pred)
        
        ### ===== 记录结果 ===== ###
        file = result_filename + '.txt'
        real_type = sample['type_pred']
        if real_type.startswith('superlative'):
            real_type = 'superlative'

        with open(file, 'a', encoding='utf-8') as f:
            f.write('<question id=' + str(id) + '>\t' + ' '.join(sample['question']) + '\n')
            f.write('<logical form id=' + str(id) + '>\t' + logical_pred + '\n')
            f.write('<parameters id=' + str(id) + '>\t' + print_entity_pred + '\n')
            f.write('<question type id=' + str(id) + '>\t' + real_type + '\n')
            f.write('==================================================\n')

        ### ===== 记录错误情况 ===== ###
        if opt.mode == 'dev':
            if type not in dict_acc:
                dict_acc[type] = [0, 0]
            if logical == logical_pred:
                x += 1
                dict_acc[type][0] += 1
                if type.startswith('superlative'):
                    dict_acc['superlative'][0] += 1
            elif all(flag):
                filename_tmp = opt.input_path.split('/')
                filename_tmp = filename_tmp[-1]
                filename_tmp = filename_tmp.split('.')
                filename_tmp = '.'.join(filename_tmp[0:-1])
                file = error_filename + '.' + filename_tmp
                with open(file, 'a', encoding='utf-8') as f:
                    f.write('<question>'.ljust(15) + ' '.join(sample['question']) + '\n')
                    f.write('<logical>'.ljust(15) + logical + '\n')
                    f.write('<prediction>'.ljust(15) + logical_pred + '\n')
                    f.write('<scores>'.ljust(15) + 'TODO3: ' + str(round(scores[0]*100, 1)) + '(' + str(round(highest[0]*100, 1)) + '); TODO1: ' \
                                                             + str(round(scores[1]*100, 1)) + '(' + str(round(highest[1]*100, 1)) + '); COVER: ' \
                                                             + str(scores[2]) + '(' + str(highest[2]) + '); LOSS: ' \
                                                             + str(round(scores[3], 6)) + '(' + str(round(highest[3], 6)) + '). \n')
                    f.write('<score-rank>'.ljust(15) + 'TODO3: ' + str(rank[0]) + '; TODO1: ' + str(rank[1]) + '; ' \
                                                     + 'COVER: ' + str(rank[2]) + '; LOSS: ' + str(rank[3]) + '.\n')
                    f.write('==================================================\n')

            dict_acc[type][1] += 1
            y += 1
            if type.startswith('superlative'):
                dict_acc['superlative'][1] += 1


    ### ===== 输出dev上的预测正确率情况 ===== ###
    if opt.mode == 'dev':
        all_accu = round(x / y * 100, 2)
        dict_acc = [(k, v[0] / v[1] * 100) for k, v in dict_acc.items()]
        dict_acc.sort(key=lambda x: x[0])

        logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
        log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.eval.result.txt'
        if opt.eval_path:
            log_file_name = os.path.join(opt.eval_path, log_file_name)
        file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
        logging.root.addHandler(file_handler)
        logger = logging.getLogger(__name__)

        logger.info(opt)
        logger.info('overall accuracy: {all: 2.2f}%\n'.format(all=all_accu))
        for d in dict_acc:
            logger.info(d[0] + ' accuracy: {acc: 2.2f}%'.format(type=d[0], acc=round(d[1], 2)))
