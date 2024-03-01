import os
from datetime import datetime
from collections import OrderedDict
from preprocess.ICU_class import Event, ColContent
import pandas
import csv
import pickle
import pandas as pd
import numpy as np
from preprocess.icuclass_gen import icu_class_gen
from preprocess.preprocess_utils import *

from transformers import BertTokenizer, BertModel
from collections import OrderedDict
from transformers import AutoTokenizer
class EHRParser:
    pid_col = 'pid'
    adm_id_col = 'adm_id'
    adm_time_col = 'adm_time'
    cid_col = 'cid'

    def __init__(self, path):
        self.path = path

        self.skip_pid_check = False

        self.patient_admission = None
        self.admission_codes = None
        self.event_dict = None
        self.admission_procedures = None
        self.admission_medications = None
        self.code2first_level_dx = None
        self.code2single_dx = None
        self.admDxMap = None
        self.admDxMap_ccs = None
        self.admDxMap_ccs_cat1 = None
        self.seqs = None
        self.seqs_ccs = None
        self.seqs_ccs_cat1 = None

        self.parse_fn = {'d': self.set_diagnosis}

    def set_admission(self):
        raise NotImplementedError

    def set_diagnosis(self):
        raise NotImplementedError

    @staticmethod
    def to_standard_icd9(code: str):
        raise NotImplementedError

    def parse_admission(self):
        #print('parsing the csv file of admission ...')
        filename, cols, converters = self.set_admission() #file=ADMISSIONS.csv cols=<class 'dict'>: {'pid': 'SUBJECT_ID', 'adm_id': 'HADM_ID', 'adm_time': 'ADMITTIME'} converters=<class 'dict'>: {'SUBJECT_ID': <class 'int'>, 'HADM_ID': <class 'int'>, 'ADMITTIME': <function Mimic3Parser.set_admission.<locals>.<lambda> at 0x7f8928f26940>}
        admissions = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        #print("parse_admission filepath:",os.path.join(self.path, filename))
        admissions = self._after_read_admission(admissions, cols) #将入院起始年份大于2012的取出来mimci-iii不做这样的处理，只有mimic-4做这样的处理
        all_patients = OrderedDict() #OrderDict([(pid,[{'adm_id':...,'adm_time':...}]),(pid,[{'adm_id':...,'adm_time':...}]),()])
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print(' \r\t%d in %d rows' % (i + 1, len(admissions)), end='')
            pid, adm_id, adm_time = row[cols[self.pid_col]], row[cols[self.adm_id_col]], row[cols[self.adm_time_col]]
            if pid not in all_patients:
                all_patients[pid] = []
            admission = all_patients[pid]
            admission.append({self.adm_id_col: adm_id, self.adm_time_col: adm_time})
        #print('\r\t%d in %d rows' % (len(admissions), len(admissions)))

        patient_admission = OrderedDict() #patient_admission将admission>=2的按照adm_time时间排序
        for pid, admissions in all_patients.items():
            if len(admissions) >= 2:
                patient_admission[pid] = sorted(admissions, key=lambda admission: admission[self.adm_time_col])
        #orderDict([(pid,[{'adm_id':...,'adm_time':...}]),(pid,[{'adm_id':...,'adm_time':...}]),()])
        #patient_admission = patient_admission[:len(patient_admission)]
        patient_admission_hadm_id = []
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                patient_admission_hadm_id.append(adm_id)
        #patient_admission_hadm_id_numpy = np.array(patient_admission_hadm_id)

        self.patient_admission = patient_admission

    def LabelsForData(self,ccs_multi_dx_file, ccs_single_dx_file):
        ccs_multi_dx_df = pd.read_csv(ccs_multi_dx_file, header=0, dtype=object)
        ccs_single_dx_df = pd.read_csv(ccs_single_dx_file, header=0, dtype=object)
        code2single_dx = {}
        code2first_level_dx = {}
        for i, row in ccs_multi_dx_df.iterrows():
            # print(row)
            code = row[0][1:-1].strip()
            level_1_cat = row[1][1:-1].strip()
            code2first_level_dx[code] = level_1_cat

        for i, row in ccs_single_dx_df.iterrows():
            code = row[0][1:-1].strip()
            single_cat = row[1][1:-1].strip()
            code2single_dx[code] = single_cat
        self.code2first_level_dx = code2first_level_dx
        self.code2single_dx = code2single_dx

    def _after_read_admission(self, admissions, cols):
        return admissions
    '''
    concepts读取diagnoses_icd.csv文件中的subject_id、ham_id、icd_code、icd_version列
    返回的result是subject_id在patient_admission中对应的code
    格式：OrderDict([(subject_id,['code','code'...]),(subject_id,['code','code'...]),...])
    '''
    def _parse_concept(self, concept_type):
        assert concept_type in self.parse_fn.keys()
        filename, cols, converters = self.parse_fn[concept_type]() #self.parse_fn = {'d': self.set_diagnosis}
        concepts = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        print("parse_diagnoses filepath",os.path.join(self.path, filename))
        concepts = self._after_read_concepts(concepts, concept_type, cols) #DtaFrame subject_id ham_id icd_code icd_version
        result = OrderedDict()

        admDxMap = {}
        admDxMap_ccs = {}
        admDxMap_ccs_cat1 = {}
        infd = open(os.path.join(self.path, filename), 'r')
        infd.readline()
        for line in infd:
            tokens = line.strip().split(',')
            '''
            在mimic-iv中tokens[0]是subject_id，tokens[1]是hadm_id, tokens[3]是icd_code
            在mimic-iii中tokens[1]是subject_id，tokens[2]是hadm_id, tokens[4]是icd_code
            '''
            pid = int(tokens[1])
            adm_id = int(tokens[2])
            if pid in self.patient_admission.keys():
                code = tokens[4][1:-1]
                #code = tokens[4] #mimic4 不需要[1:-1]
                if code == '':
                    continue
                dxStr = 'D_' + code
                dxStr_ccs_single = 'D_' + self.code2single_dx[code]  # 是从ccs_single_dx_tool_2015.csv中得到的
                dxStr_ccs_cat1 = 'D_' + self.code2first_level_dx[code]  # cat1是从ccs_multi_dx_tool_2015.csv中得到的

                if adm_id not in result:
                    result[adm_id] = []
                codes = result[adm_id]
                codes.append(dxStr)
                if adm_id in admDxMap:
                    admDxMap[adm_id].append(dxStr)
                else:
                    admDxMap[adm_id] = [dxStr]
                if adm_id in admDxMap_ccs:
                    admDxMap_ccs[adm_id].append(dxStr_ccs_single)
                else:
                    admDxMap_ccs[adm_id] = [dxStr_ccs_single]
                if adm_id in admDxMap_ccs_cat1:
                    admDxMap_ccs_cat1[adm_id].append(dxStr_ccs_cat1)
                else:
                    admDxMap_ccs_cat1[adm_id] = [dxStr_ccs_cat1]
        # for i, row in concepts.iterrows():
        #     if i % 100 == 0:
        #         print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
        #     pid = row[cols[self.pid_col]]
        #     '''
        #     self.patient_admission
        #     orderDict([(pid,[{'adm_id':...,'adm_time':...}]),(pid,[{'adm_id':...,'adm_time':...}]),()])
        #     '''
        #     if self.skip_pid_check or pid in self.patient_admission:
        #         adm_id, code = row[cols[self.adm_id_col]], row[cols[self.cid_col]]
        #         if code == '':
        #             continue
        #         if adm_id not in result:
        #             result[adm_id] = []
        #         codes = result[adm_id]
        #         codes.append(code)
        # df = pd.DataFrame(list(code_num_dict.items()), columns=["Key", "Value"])
        # df.to_csv("/home/ubuntu/Zhang/integrate/my_dict.csv", index=False)

        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        return result, admDxMap, admDxMap_ccs, admDxMap_ccs_cat1  #result

    def _after_read_concepts(self, concepts, concept_type, cols):
        return concepts

    def parse_diagnoses(self):
        print('parsing csv file of diagnosis ...')
        self.admission_codes, self.admDxMap, self.admDxMap_ccs, self.admDxMap_ccs_cat1 = self._parse_concept('d')



    def parse_event(self):
        print(f'mimic class generation target list : ')
        config = {
            'Table': {
                'mimic3':
                    [
                        {'table_name': 'LABEVENTS', 'time_column': 'CHARTTIME', 'table_type': 'lab',
                         'time_excluded': ['ENDTIME'], 'id_excluded': ['ROW_ID', 'ICUSTAY_ID', 'SUBJECT_ID']
                         },
                        {'table_name': 'PRESCRIPTIONS', 'time_column': 'STARTDATE', 'table_type': 'med',
                         'time_excluded': ['ENDDATE'],
                         'id_excluded': ['GSN', 'NDC', 'ROW_ID', 'ICUSTAY_ID', 'SUBJECT_ID']
                         },
                        {'table_name': 'INPUTEVENTS_MV', 'time_column': 'STARTTIME', 'table_type': 'inf',
                         'time_excluded': ['ENDTIME', 'STORETIME'],
                         'id_excluded': ['CGID', 'ORDERID', 'LINKORDERID', 'ROW_ID', 'ICUSTAY_ID', 'SUBJECT_ID']
                         },
                        {'table_name': 'INPUTEVENTS_CV', 'time_column': 'CHARTTIME', 'table_type': 'inf',
                         'time_excluded': ['STORETIME'],
                         'id_excluded': ['CGID', 'ORDERID', 'LINKORDERID', 'ROW_ID', 'ICUSTAY_ID', 'SUBJECT_ID']
                         }
                    ],
                'eicu':
                    [
                        {'table_name': 'lab', 'time_column': 'labresultoffset', 'table_type': 'lab',
                         'time_excluded': ['labresultrevisedoffset'], 'id_excluded': ['labid']
                         },
                        {'table_name': 'medication', 'time_column': 'drugstartoffset', 'table_type': 'med',
                         'time_excluded': ['drugorderoffset, drugstopoffset'],
                         'id_excluded': ['medicationid', 'GTC', 'drughiclseqno']
                         },
                        {'table_name': 'infusionDrug', 'time_column': 'infusionoffset', 'table_type': 'inf',
                         'time_excluded': None, 'id_excluded': None
                         }
                    ],
                'mimic4':
                    [
                        {'table_name': 'labevents', 'time_column': 'CHARTTIME', 'table_type': 'lab',
                         'time_excluded': ['STORETIME', 'INTIME', 'OUTTIME', 'common_time'],
                         'id_excluded': ['ICUSTAY_ID', 'SUBJECT_ID', 'SPECIMEN_ID']
                         },
                        {'table_name': 'prescriptions', 'time_column': 'STARTTIME', 'table_type': 'med',
                         'time_excluded': ['STOPTIME', 'INTIME', 'OUTTIME', 'common_time'],
                         'id_excluded': ['GSN', 'NDC', 'ICUSTAY_ID', 'SUBJECT_ID', 'PHARMACY_ID']
                         },
                        {'table_name': 'inputevents', 'time_column': 'STARTTIME', 'table_type': 'inf',
                         'time_excluded': ['ENDTIME', 'STORETIME', 'INTIME', 'OUTTIME', 'common_time'],
                         'id_excluded': ['ORDERID', 'LINKORDERID', 'ICUSTAY_ID', 'SUBJECT_ID']
                         }

                    ]
            },
            'selected': {
                'mimic3': {
                    'LABEVENTS': {
                        'HADM_ID': 'ID',
                        'CHARTTIME': 'TIME',
                        'ITEMID': 'code',
                        'VALUENUM': 'value',
                        'VALUEUOM': 'uom',
                    },
                    'PRESCRIPTIONS': {
                        'HADM_ID': 'ID',
                        'STARTDATE': 'TIME',
                        'DRUG': 'code',
                        'ROUTE': 'route',
                        'PROD_STRENGTH': 'prod',
                        'DOSE_VAL_RX': 'value',
                        'DOSE_UNIT_RX': 'uom',
                    },
                    'INPUTEVENTS_MV': {
                        'HADM_ID': 'ID',
                        'STARTTIME': 'TIME',
                        'ITEMID': 'code',
                        'RATE': 'value',
                        'RATEUOM': 'uom',
                    },
                    'INPUTEVENTS_CV': {
                        'HADM_ID': 'ID',
                        'CHARTTIME': 'TIME',
                        'ITEMID': 'code',
                        'RATE': 'value',
                        'RATEUOM': 'uom',
                    }
                },

                'eicu': {
                    'lab': {
                        'patientunitstayid': 'ID',
                        'labresultoffset': 'TIME',
                        'labname': 'code',
                        'labresult': 'value',
                        'labmeasurenamesystem': 'uom'
                    },
                    'medication': {
                        'patientunitstayid': 'ID',
                        'drugstartoffset': 'TIME',
                        'drugname': 'code',
                        'routeadmin': 'route',
                    },
                    'infusionDrug': {
                        'patientunitstayid': 'ID',
                        'infusionoffset': 'TIME',
                        'drugname': 'code',
                        'infusionrate': 'value'
                    }
                },
                'mimic4': {
                    'labevents': {
                        'hadm_id': 'ID', #HADM_ID
                        #'charttime': 'TIME', #CHARTTIME
                        'itemid': 'code', #ITEMID
                        'valuenum': 'value', #VALUENUM
                        'valueuom': 'uom', #VALUEUOM
                    },
                    'prescriptions': {
                        'hadm_id': 'ID',
                        #'startdate': 'TIME', #STARTDATE
                        'drug': 'code', #DRUG
                        'prod_strength': 'prod', #PROD_STRENGTH
                        'dose_val_rx': 'value', #DOSE_VAL_RX
                        'dose_unit_rx': 'uom', #DOSE_UNIT_RX
                    },
                    'inputevents': {
                        'hadm_id': 'ID', #hadm_id
                        #'starttime': 'TIME', #starttime
                        'itemid': 'code', #drug
                        'rate': 'value', #RATE
                        'rateuom': 'uom', #RATEUOM
                    },
                }

            },
            'DICT_FILE': {
                'mimic3': {
                    'LABEVENTS': ['D_LABITEMS', 'ITEMID'],
                    'INPUTEVENTS_CV': ['D_ITEMS', 'ITEMID'],
                    'INPUTEVENTS_MV': ['D_ITEMS', 'ITEMID']
                },

                'eicu': {
                },

                'mimic4': {
                    'labevents': ['d_labitems', 'itemid'],
                    'inputevents': ['d_items', 'itemid'],
                },

            },
            'ID': {
                'mimic3':
                    'HADM_ID',

                'eicu':
                    'patientunitstayid',

                'mimic4':
                    'hadm_id'#HADM_ID
            },

        }
        #tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/Zhang/Chet_events_LSTM_cuda/Chet_events/Bio_ClinicalBERT")
        # input_tok = tokenizer.encode('INPUTEVENTS')[1:-1]
        for src in ['mimic3']:  #参数
            print(f'mimic gen start : {src} ')
            # tokenize = tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/zx/Chet_events/Bio_ClinicalBERT")
            # model = BertModel.from_pretrained('/home/ubuntu/zx/Chet_events/Bio_ClinicalBERT')
            icu_dict = icu_class_gen(src, config)
            #icu_dict = icu_dict.dict_slice(icu_dict, 0, 5)
            event_dict = OrderedDict()
            for pid, icu in icu_dict.items():
                sum = []
                prescription = []
                lab = []
                inputs = []

                if icu.events != []:
                    for event in icu.events:
                        #MIMIC-III
                        if event.table == 'LABEVENTS':
                            # lab_tok = event.table_tok_id
                            # lab = lab + event.content_value
                            lab = lab + event.content_value
                        if event.table == 'PRESCRIPTIONS':
                            # prescription_tok = event.table_tok_id
                            # prescription = prescription + event.content_value
                            prescription = prescription + event.content_value
                        if event.table == ('INPUTEVENTS_MV' or 'INPUTEVENTS_CV'):
                            # tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/Zhang/EVENTS/Bio_ClinicalBERT")
                            # input_tok = tokenizer.encode('INPUTEVENTS')[1:-1]
                            # inputs = inputs + event.content_value
                            inputs = inputs + event.content_value
                        '''
                        #MIMIC-IV
                        if event.table == 'labevents':
                            #lab_tok = event.table_tok_id
                            #lab = lab + event.content_value
                            lab = lab + event.content_value
                        if event.table == 'prescriptions':
                            #prescription_tok = event.table_tok_id
                            #prescription = prescription + event.content_value
                            prescription = prescription + event.content_value
                        if event.table == 'inputevents':
                            #tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/zx/Chet_events/Bio_ClinicalBERT")
                            #input_tok = tokenizer.encode('inputevents')[1:-1]
                            #inputs = inputs + event.content_value
                            inputs = inputs + event.content_value
                        '''


                if lab != []:
                    #lab = tokenizer.encode('LABEVENTS')[1:-1] #MIMIC-III
                    #lab = [101] + lab_tok + [102] + lab
                    #lab_tok = event.table_tok_id
                    #lab = lab + event.content_value [[],[]]
                    if type(lab) is not list:
                        lab = list(lab)
                    sum.append(lab)
                if prescription != []:
                    #prescription = [101] + prescription_tok + [102] + prescription
                    if type(prescription) is not list:
                        prescription = list(prescription)
                    sum.append(prescription)
                if inputs != []:
                #inputs = [101] + input_tok + [102] + inputs
                    if type(inputs) is not list:
                        inputs = list(inputs)
                    sum.append(inputs)

                #sum = sum.append(lab+prescription+inputs)
                if (pid not in event_dict) and (sum != []):
                    #print('pid not in event_dict')
                    event_dict[pid] = sum
        #{ham_id:[[enent1],[event2]]}
        self.event_dict = event_dict

        print("111")

    def calibrate_patient_by_admission(self):
        #print('calibrating patients by admission ...') #weishen
        del_pids = []
        del_event_pids = []
        for pid, admissions in self.patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.admission_codes:
                    break
            else:
                continue
            del_pids.append(pid)

        for event_pid, admissions in self.patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.event_dict.keys():
                    break
            else:
                continue
            del_event_pids.append(event_pid)  # 7537

        for pid in del_pids:
            admissions = self.patient_admission[pid]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                for concepts in [self.admission_codes]:
                    if adm_id in concepts:
                        del concepts[adm_id]
                    if adm_id in self.admDxMap:
                        del self.admDxMap[adm_id]
                    if adm_id in self.admDxMap_ccs:
                        del self.admDxMap_ccs[adm_id]
                    if adm_id in self.admDxMap_ccs_cat1:
                        del self.admDxMap_ccs_cat1[adm_id]
            del self.patient_admission[pid]

        for event_pids in del_event_pids:
            if event_pids in self.patient_admission:
                admissions = self.patient_admission[event_pids]
                for admission in admissions:
                    adm_id = admission[self.adm_id_col]
                    if adm_id in self.event_dict.keys():
                        del self.event_dict[adm_id]
                # del self.patient_admission[event_pids]

    def calibrate_admission_by_patient(self):
        print('calibrating admission by patients ...')
        adm_id_set = set()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id_set.add(admission[self.adm_id_col])
        del_adm_ids = [adm_id for adm_id in self.admission_codes if adm_id not in adm_id_set]
        del_event_adm_ids = [adm_id for adm_id in self.event_dict.keys() if adm_id not in adm_id_set]

        for adm_id in del_adm_ids:
            del self.admission_codes[adm_id], self.admDxMap[adm_id], self.admDxMap_ccs[adm_id], self.admDxMap_ccs_cat1[adm_id]

        for adm_event_id in del_event_adm_ids:
            if adm_event_id in self.event_dict:
                del self.event_dict[adm_event_id]

    def code_num(self):
        num_dict = {}
        for pid, admissions in self.patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id in self.admission_codes:
                    for codes in self.admission_codes[adm_id]:
                        if codes not in num_dict:
                            num_dict[codes] = 0
                        num_dict[codes] = num_dict[codes]+1
        return num_dict

    def sample_patients(self, sample_num, seed):
        np.random.seed(seed)
        keys = list(self.patient_admission.keys())
        #selected_pids = np.random.choice(keys, sample_num, False)
        selected_pids = keys
        self.patient_admission = {pid: self.patient_admission[pid] for pid in selected_pids}
        admission_codes = dict()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                admission_codes[adm_id] = self.admission_codes[adm_id]
        self.admission_codes = admission_codes

    def adm_map_seq(self):
        pidSeqMap = {}
        pidSeqMap_ccs = {}
        pidSeqMap_ccs_cat1 = {}
        for pid , admissions in self.patient_admission.items():
            new_admIdList = []
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id in self.admDxMap:
                    new_admIdList.append(adm_id)
            sortedList = sorted([self.admDxMap[admId] for admId in new_admIdList])
            pidSeqMap[pid] = sortedList

            sortedList_ccs = sorted([self.admDxMap_ccs[admId] for admId in new_admIdList])
            pidSeqMap_ccs[pid] = sortedList_ccs
            sortedList_ccs_cat1 = sorted([self.admDxMap_ccs_cat1[admId] for admId in new_admIdList])
            pidSeqMap_ccs_cat1[pid] = sortedList_ccs_cat1
        return pidSeqMap, pidSeqMap_ccs, pidSeqMap_ccs_cat1

    def pidSeqMap_to_seqs(self, pidSeqMap, pidSeqMap_ccs, pidSeqMap_ccs_cat1):
        seqs = []
        for pidSeqMap_pid, pidSeqMap_values in pidSeqMap.items():
            seq = []
            for value in pidSeqMap_values:
                seq.append(value)
            seqs.append(seq)
        self.seqs = seqs
        seqs_ccs = []  # 每一次就诊的DxMap_ccs
        for pidSeqMap_ccs_pid, pidSeqMap_ccs_values in pidSeqMap_ccs.items():
            seq = []
            for value in pidSeqMap_ccs_values:
                seq.append(value)
            seqs_ccs.append(seq)
        self.seqs_ccs = seqs_ccs
        seqs_ccs_cat1 = []  # 每一次就诊的DxMap_ccs
        for pidSeqMap_ccs_cat1_pid, pidSeqMap_ccs_cat1_values in pidSeqMap_ccs_cat1.items():
            seq = []
            for value in pidSeqMap_ccs_cat1_values:
                seq.append(value)
            seqs_ccs_cat1.append(seq)
        self.seqs_ccs_cat1 = seqs_ccs_cat1

    def parse(self, sample_num=None, seed=6669):
        multi_dx_file = 'ccs/ccs_multi_dx_tool_2015.csv'
        single_dx_file = 'ccs/ccs_single_dx_tool_2015.csv'
        self.parse_admission()
        self.LabelsForData(multi_dx_file, single_dx_file)
        self.parse_diagnoses()
        self.parse_event()
        #self.patient_admission中的adm_id不在self.admission_codes中删除self.patient_admission中的对应行
        self.calibrate_patient_by_admission()
        #self.admission_codes中的adm_id不在self.patient_admission中删除self.admission_codes中的对应行
        self.calibrate_admission_by_patient()
        #取sample_num数量的数据,将self.patient_admission、self.admission_codes组成字典
        #这里检测患者就诊的疾病中是否都有稀有疾病，如果全部都有稀有疾病的话，要把对应的hadm_id删除掉
        #这段代码只是为了保证疾病是充足的

        '''
        num_dict = self.code_num() # 得到每种疾病的频率
        for pid, admissions in self.patient_admission.items():
            print("删除频率低的疾病....")
            for i, admission in enumerate(admissions):
                adm_id = admission[self.adm_id_col]
                self.admission_codes[adm_id] = [disease for disease in self.admission_codes[adm_id] if num_dict[disease]<=6]
                if self.admission_codes[adm_id] == []:
                    print("就诊为空......")
                    admissions.pop(i)
                    del self.admDxMap[adm_id]
                    del self.admDxMap_ccs[adm_id]
                    del self.admDxMap_ccs_cat1[adm_id]
                    del self.admission_codes[adm_id]
                    del self.event_dict[adm_id]
        '''

        if sample_num is not None:
            self.sample_patients(10000, seed) #对于mimic4而言
        #print('Building pid-sortedVisits mapping')
        pidSeqMap, pidSeqMap_ccs, pidSeqMap_ccs_cat1 = self.adm_map_seq()

        #print('Building strSeqs, strSeqs for CCS single-level code, strSeqs for CCS multi-level first code')
        self.pidSeqMap_to_seqs(pidSeqMap, pidSeqMap_ccs, pidSeqMap_ccs_cat1)  # 返回seq, seq_ccs, seq_ccs_cat1

        #print('Converting strSeqs to intSeqs, and making types for ccs single-level code')
        dict_ccs = {}  #
        newSeqs_ccs = []  # 每个code_css有一个索引，利用索引表示code_css
        count_code = 0
        for patient in self.seqs_ccs:  # 遍历的是每个pid
            newPatient = []
            for visit in patient:
                count_code = count_code + len(visit)
                newVisit = []
                for code in set(visit):  # set相当于去重
                    if code in dict_ccs:
                        newVisit.append(dict_ccs[code])
                    else:
                        dict_ccs[code] = len(dict_ccs)
                        newVisit.append(dict_ccs[code])
                newPatient.append(newVisit)
            newSeqs_ccs.append(newPatient)

        #print('Building strSeqs for CCS multi-level first code')
        seqs_ccs_cat1 = []
        for pid, visits in pidSeqMap_ccs_cat1.items():
            seq = []
            for visit in visits:
                seq.append(visit)
            seqs_ccs_cat1.append(seq)

        #print('Converting strSeqs to intSeqs, and making types for ccs multi-level first level code')
        dict_ccs_cat1 = {}  ##每个code_cat1有一个索引，利用索引表示code_cat1
        newSeqs_ccs_cat1 = []
        for patient in seqs_ccs_cat1:
            newPatient = []
            for visit in patient:
                newVisit = []
                for code in set(visit):
                    if code in dict_ccs_cat1:
                        newVisit.append(dict_ccs_cat1[code])
                    else:
                        dict_ccs_cat1[code] = len(dict_ccs_cat1)
                        newVisit.append(dict_ccs_cat1[code])
                newPatient.append(newVisit)
            newSeqs_ccs_cat1.append(newPatient)
        '''
        self.admission_codes
        '''#list(od.keys())
        #np.savez('/home/ubuntu/zx/Chet_events_LSTM/Chet_events', list(self.patient_admission.keys()))
        #print("patient_hadmid已保存")

        # num_dict = self.code_num()
        # df = pd.DataFrame(list(num_dict.items()), columns=["Key", "Value"])
        # df.to_csv("num_dict.csv", index=False)
        return self.patient_admission, self.admission_codes, self.event_dict, \
               self.admDxMap, self.admDxMap_ccs, self.admDxMap_ccs_cat1, self.seqs_ccs, self.seqs_ccs_cat1, self.seqs, dict_ccs, dict_ccs_cat1




class Mimic3Parser(EHRParser):
    def set_admission(self):
        filename = 'ADMISSIONS.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.adm_time_col: 'ADMITTIME'}
        converter = {
            'SUBJECT_ID': int,
            'HADM_ID': int,
            'ADMITTIME': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'DIAGNOSES_ICD.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.cid_col: 'ICD9_CODE'}
        converter = {'SUBJECT_ID': int, 'HADM_ID': int, 'ICD9_CODE': Mimic3Parser.to_standard_icd9}
        return filename, cols, converter

    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code


class Mimic4Parser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.icd_ver_col = 'icd_version'
        self.icd_map = self._load_icd_map() #字典格式；['icd-10':'icd-9']
        self.patient_year_map = self._load_patient()#字典格式;[subject_id病人病号:anchor_year-anchor_year_group[:4]这个也不知道啥意思 ]

    def _load_icd_map(self):
        print('loading ICD-10 to ICD-9 map ...')
        filename = 'icd10-icd9.csv'
        cols = ['ICD10', 'ICD9']
        converters = {'ICD10': str, 'ICD9': str}
        icd_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        icd_map = {row['ICD10']: row['ICD9'] for _, row in icd_csv.iterrows()}
        return icd_map

    def _load_patient(self):
        print('loading patients anchor year ...')
        filename = 'patients.csv'
        cols = ['subject_id', 'anchor_year', 'anchor_year_group']
        converters = {'subject_id': int, 'anchor_year': int, 'anchor_year_group': lambda cell: int(str(cell)[:4])}
        patient_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        patient_year_map = {row['subject_id']: row['anchor_year'] - row['anchor_year_group']
                            for i, row in patient_csv.iterrows()}
        return patient_year_map

    def set_admission(self):
        filename = 'admissions.csv'
        cols = {self.pid_col: 'subject_id', self.adm_id_col: 'hadm_id', self.adm_time_col: 'admittime'}
        converter = {
            'subject_id': int,
            'hadm_id': int,
            'admittime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnoses_icd.csv'
        cols = {
            self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            self.cid_col: 'icd_code',
            self.icd_ver_col: 'icd_version'
        }
        converter = {'subject_id': int, 'hadm_id': int, 'icd_code': str, 'icd_version': int}
        return filename, cols, converter

    def _after_read_admission(self, admissions, cols):
        print('\tselecting valid admission ...')
        valid_admissions = []
        n = len(admissions)
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t\t%d in %d rows' % (i + 1, n), end='')
            pid = row[cols[self.pid_col]]
            year = row[cols[self.adm_time_col]].year - self.patient_year_map[pid]
            if year > 2012:
                valid_admissions.append(i)
        print('\r\t\t%d in %d rows' % (n, n))
        print('\t\tremaining %d rows' % len(valid_admissions))
        return admissions.iloc[valid_admissions]

    def _after_read_concepts(self, concepts, concept_type, cols):
        print('\tmapping ICD-10 to ICD-9 ...')
        n = len(concepts)
        if concept_type == 'd':
            def _10to9(i, row):
                if i % 100 == 0:
                    print('\r\t\t%d in %d rows' % (i + 1, n), end='')
                cid = row[cid_col]
                if row[icd_ver_col] == 10:
                    if cid not in self.icd_map:
                        code = self.icd_map[cid + '1'] if cid + '1' in self.icd_map else ''
                    else:
                        code = self.icd_map[cid]
                    if code == 'NoDx':
                        code = ''
                else:
                    code = cid
                return Mimic4Parser.to_standard_icd9(code)
            #cols:<class 'dict'>: {'pid': 'subject_id', 'adm_id': 'hadm_id', 'cid': 'icd_code', 'icd_version': 'icd_version'}
            #cid_col:'icd_code'
            cid_col, icd_ver_col = cols[self.cid_col], self.icd_ver_col

            col = np.array([_10to9(i, row) for i, row in concepts.iterrows()])
            print('\r\t\t%d in %d rows' % (n, n))
            concepts[cid_col] = col
        return concepts

    @staticmethod
    def to_standard_icd9(code: str):
        return Mimic3Parser.to_standard_icd9(code)



