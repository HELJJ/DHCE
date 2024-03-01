import random

import numpy as np

from preprocess.parse_csv import EHRParser
from transformers import BertTokenizer, BertModel
from collections import OrderedDict
from transformers import AutoTokenizer
import math
import torch
from torch import nn
from transformers import logging
logging.set_verbosity_error()
def split_patients(patient_admission, admission_codes, code_map, train_num, test_num, seed=6669):
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission[EHRParser.adm_id_col]] #adm_id_col:adm_id
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    # valid_num = len(patient_admission) - train_num - test_num
    # train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    # valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    # test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    #train_num = int(len(common_pids))
    #train_pids = np.array(len(common_pids)*0.8)

    '''
    valid_num = int(len(common_pids)*0.1)
    #valid_pids = np.array(list(set(patient_admission.keys()).difference(train_pids)))
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]

    train_num = int(len(common_pids)*0.8)
    train_pids = np.array(random.sample(common_pids, train_num))
    #valid_pids = np.array(random.sample(common_pids,valid_num))
    valid_pids =  np.array(random.sample(common_pids - set(train_pids), valid_num))
    # train_pids_K_list = common_pids
    # valid_pids_K_list =
    # test_pids_K_list =

    return train_pids, valid_pids, test_pids
    '''
    valid_num = len(patient_admission) - train_num - test_num
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    for test_pid_1 in test_pids: #2-3
        if len(patient_admission[test_pid_1]) > 1 and len(patient_admission[test_pid_1]) < 4:
            continue
        else:
            test_pids_23 = np.delete(test_pids, np.where(test_pids == test_pid_1))
    np.save("/media/qlunlp/86f9f306-e54a-46ad-b338-2b9f8634686f/qluai/Zhang/Chet_integrate/test_pids_visit23.npy", test_pids_23)

    for test_pid_2 in test_pids: #3-4
        if len(patient_admission[test_pid_2]) > 2 and len(patient_admission[test_pid_2]) < 5:
            continue
        else:
            test_pids_34 = np.delete(test_pids, np.where(test_pids == test_pid_2))
    np.save("/media/qlunlp/86f9f306-e54a-46ad-b338-2b9f8634686f/qluai/Zhang/Chet_integrate/test_pids_visit34.npy", test_pids_34)

    for test_pid_3 in test_pids: #4-5
        if len(patient_admission[test_pid_3]) > 3 and len(patient_admission[test_pid_3]) < 6:
            continue
        else:
            test_pids_45 = np.delete(test_pids, np.where(test_pids == test_pid_3))
    np.save("/media/qlunlp/86f9f306-e54a-46ad-b338-2b9f8634686f/qluai/Zhang/Chet_integrate/test_pids_45.npy", test_pids_45)

    for test_pid_4 in test_pids: #5-6
        if len(patient_admission[test_pid_4]) > 4 and len(patient_admission[test_pid_4]) < 7:
            continue
        else:
            test_pids_56 = np.delete(test_pids, np.where(test_pids == test_pid_4))
    np.save("/media/qlunlp/86f9f306-e54a-46ad-b338-2b9f8634686f/qluai/Zhang/Chet_integrate/test_pids_56.npy", test_pids_56)
    return train_pids, valid_pids, test_pids


def build_code_xy(pids, admDxMap, patient_admission, admission_codes_encoded, max_admission_num, code_num, event_dict):

    n = len(pids)
    x = np.zeros((n, max_admission_num, code_num), dtype=bool)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n,), dtype=int)
    patient_clinicalevents = []
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            x[i, k, codes] = 1
        codes = np.array(admission_codes_encoded[admissions[-1][EHRParser.adm_id_col]])
        y[i, codes] = 1
        lens[i] = len(admissions) - 1
    #tokenize = tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/Zhang/EVENTS/Bio_ClinicalBERT")
    #model = BertModel.from_pretrained('/home/ubuntu/Zhang/Chet_events/Bio_ClinicalBERT')
    patients_visit = []#[patient[visit1[event[]],visit2[event[]],visit3[]], []]
    for i, pid in enumerate(pids):
        #print('患者pid:{}'.format(pid))
        if pid in patient_admission:
            admissions = patient_admission[pid]
            visits_event = []
            for k, admission in enumerate(admissions[:-1]):#每一次就诊150348
                one_visit = []
                if admission[EHRParser.adm_id_col] in event_dict:
                    events = event_dict[admission[EHRParser.adm_id_col]] #对于一次就诊(adm_id)对应的事件(可能会有多种事件)

                    # for event in events:
                    #     one_visit.append(events)
                    '''
                print('\n')
                print('{} {} start:'.format(pid, admission[EHRParser.adm_id_col]))
                for item in events:#
                    item_lens = len(item)
                    if item_lens>512:
                        input_ids = torch.tensor(item[:512])
                        mask_len = [1] * item_lens
                        attention_mask = torch.tensor(mask_len[:512])
                    else:
                        mask_len = [1] * item_lens
                        input_ids = torch.tensor(item)
                        attention_mask = torch.tensor(mask_len)
                    # (1,768)nsor a (88) must match the size of tensor b (41) at non-singleton dim
                    print('{} {} end:'.format(pid,admission[EHRParser.adm_id_col]))
                    print('input_ids:{}',format(input_ids))
                    print('attention_mask:{}'.format(attention_mask))
                    output = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
                    print('{} {} model output:'.format(pid,admission[EHRParser.adm_id_col]))
                    one_visit_event.append(output[1])
                    print('{} {} one_visit_event output:'.format(pid, admission[EHRParser.adm_id_col]))
                #stack_outputs = torch.vstack(one_visit_event)
                #attention = DotProductAttention(768, 32)15 24
                #one_visit_evet_rep = attention(stack_outputs) #evet_rep：在某次就诊中多种事件的表示
                visits_event_rep.append(one_visit_event)#
                print('{} {} visits_event_rep output:'.format(pid,admission[EHRParser.adm_id_col]))
            patient_clinicalevents.append(visits_event_rep)
            print('patient_clinicalevents output:')
            print('\n')
            '''
                else:
                    print('{} not in the event_dict'.format(admission[EHRParser.adm_id_col]))
                    events = []
                #visits_event.append(one_visit)
                visits_event.append(events)
            patients_visit.append(visits_event)

        else:
            print('{} not in patient_admission'.format(pid))
    #patient_clinicalevents = np.array(patient_clinicalevents)

    know_code = []
    for i, pid in enumerate(pids):
        know_code_pid_code = []
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            ccs_code = admDxMap[admission[EHRParser.adm_id_col]]
            know_code_pid_code.append(ccs_code)
        know_code.append(know_code_pid_code)
    print('\r\t%d / %d' % (len(pids), len(pids))) #[patient[visits[visit1],[visit2],..]]

    return x, y, lens, patients_visit, know_code#x是;lens 患者id对应的就诊次数


# def build_heart_failure_y(hf_prefix, codes_y, code_map):
#     hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
#     hfs = np.zeros((len(code_map),), dtype=int)
#     hfs[hf_list] = 1
#     hf_exist = np.logical_and(codes_y, hfs)
#     y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
#     return y

class DotProductAttention(nn.Module):
    def __init__(self, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, x):
        t = self.dense(x)
        vu = torch.matmul(t, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)
        return output
