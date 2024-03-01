import pandas as pd
import os
import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer
from preprocess.ICU_class import Event, ColContent
from datetime import date
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from preprocess.preprocess_utils_iv import *
from joblib import Parallel, delayed
import pdb
# from easydict import EasyDict as edict
from functools import partial
import random
from operator import itemgetter
import pickle
from transformers import AutoTokenizer
current_path = os.path.dirname(__file__)

# tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/Zhang/Chet_events_LSTM_cuda/Chet_events/Bio_ClinicalBERT")

def Table2event_multi(row, columns, table_type, column_select):
    pid_Table2event_multi = int(row['ID'])  # pid->hadm_id
    pid = int(row['ID'])
    table_type_Table2event_multi = table_type
    # time = row['TIME']
    value_Table2event_multi = round(row['value'], 4) if column_select == True and type(row['value']) in ['float',
                                                                                                         'int'] else 0

    event = Event(
        src='mimic',
        pid=pid,
        # event_time = time,
        table=table_type,
        # table_tok_id = tokenizer.encode(table_type)[1:-1],
        columns=columns,
        value=round(row['value'], 4) if column_select == True and type(row['value']) in ['float', 'int'] else 0

    )

    try:
        col_contents = [ColContent(
            table=table_type,
            col=col,
            content=round_digits(row[col]),
        ) for col in columns if row[col] != ' ']
        content_value = []
        for item in col_contents:
            if item.col == 'code':
                content_value = content_value + item.content_tok_id
            # if item.col == 'value':
            #     content_value = content_value + item.content_tok_id
        event.col_contents.extend(col_contents)
        event.content_value.extend(content_value)
        if column_select == True:
            event.column_textizing()
        return event

    except:
        print('pid:', pid)
        print('df[pid]:')
        # print('col:',col)


# multi event append
def table2event(
        config,
        src: str,
        icu: pd.DataFrame,
        table_dict: Dict[str, str],
        drop_cols_dict: Dict[str, List],
        column_select
):
    df_dict = {}
    event_dict = {}
    for table_name, df_path in table_dict.items():
        print("table_name:", table_name)
        df = pd.read_csv(df_path, skiprows=lambda i: i > 0 and random.random() > 0.05)
        print("event_df filepath:", df_path)

        # mimic-iv
        if 'value' in df.columns.values.tolist():
            df.rename(columns={'value': 'Value'}, inplace=True)

        print(f'Table loaded from {df_path} ..')
        print('src = ', src, '\n table = ', table_name)

        # preprocess start-----------------------------
        df = filter_ICU_ID(icu, df, config, src)
        df = name_dict(df, table_name, config, src, column_select)  # item_id与label之间的映射
        if column_select == True:  # 选择某些列
            df = column_select_filter(df, config, src, table_name)
        elif column_select == False:
            df = used_columns(df, table_name, config, src)
            df = ID_rename(df, table_name, config, src)
        columns = df.columns.drop(drop_cols_dict[src])
        print('columns:', columns)
        # df = ICU_merge_time_filter(icu, df, src) #关于时间的过滤
        # preprocess finish---------------------------

        # fill na
        df.fillna(' ', inplace=True)
        df.replace('nan', ' ', inplace=True)
        df.replace('n a n', ' ', inplace=True)
        # df = df.drop(columns=['TIME'])#mimic4 将这行注释掉
        # df for sanity check
        df_dict[table_name] = df
        import pdb
        # pdb.set_trace()
        events_list = Parallel(n_jobs=32, verbose=5)(
            delayed(Table2event_multi)(
                df.iloc[i],
                columns,
                table_name,
                column_select
            )
            for i in range(len(df))
        )
        print('generate event_list finish!')
        event_dict[table_name] = events_list
    return event_dict, df_dict


from datetime import datetime


def set_admission():
    pid_col = 'pid'
    adm_id_col = 'adm_id'
    adm_time_col = 'adm_time'
    cid_col = 'cid'
    filename = 'admissions.csv'  # mimic4是admissions.csv
    cols = {pid_col: 'subject_id',
            adm_id_col: 'hadm_id'}  # SUBJECT_ID HADM_ID在mimic4是subject_id; HADM_ID在mimic4是hadm_id
    converter = {
        'subject_id': int,
        'hadm_id': int,
        'admittime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')  # ADMITTIME 在mimic4中是admittime
    }
    return filename, cols, converter


def icu_class_gen(src, config):
    column_select = True  # true
    print('column select : ', column_select)
    # ICU cohort load
    # icu_path = os.path.join(os.path.join(
    #     args.save_path, 'preprocess', args.icu_type, 'time_gap_'+str(args.time_gap_hours),f'{src}_cohort.pkl')
    #     )
    # icu = pd.read_pickle(icu_path)
    # icu = pd.read_csv('/home/qluai/ALanZhang/Chet_1_11.12_everypatient_everyevent/data/mimic/PATIENTS.csv')
    filename, cols, converters = set_admission()  #
    # MIMIC-III
    # icu = pd.read_csv('/home/ubuntu/zx/Chet_events/data/mimic4/raw/ADMISSIONS.csv', usecols=list(cols.values()), converters=converters)
    # MIMIC-IV admissions
    icu = pd.read_csv(current_path + '/../data/mimic4/raw/admissions.csv',
                      usecols=list(cols.values()), converters=converters)
    #print("event_icu filepath:", '/home/ubuntu/Zhang/Chet_events_LSTM_cuda/Chet_events/data/mimic3/raw/ADMISSIONS.csv')
    # hadm_ids = np.load('/home/ubuntu/zx/Chet_events/patient_hadmid.npz')

    # hadm_ids_list = list(hadm_ids)
    # print("len(hadm_ids_list)", len(hadm_ids_list))
    icu.rename(columns={config['ID'][src]: 'ID'}, inplace=True)
    # print(f'ICU loaded from {icu_path}')
    # print(icu.info())

    # print("icu 删除之前：", len(icu))
    # for id in icu['ID']:
    #     if id not in hadm_ids_list:
    #         icu = icu.drop(icu[icu['ID']==id].index)
    # print("icu 删除之hou：", len(icu))

    # prepare ICU class
    # icu_dict = prepare_ICU_class(icu, src)
    # generate table_dict for table2event
    table_dict = dict()
    for idx in range(len(config['Table'][src])):
        table_name = config['Table'][src][idx]['table_name']
        df_path = os.path.join(current_path + '/../data/', 'mimic4',
                               table_name + '.csv')
        table_dict[table_name] = df_path
    print('table dict check \n', table_dict)
    drops_cols_dict = {
        'mimic3': ['ID', 'TIME'],
        'eicu': ['ID', 'TIME'],
        'mimic4': ['ID']
    }

    # Generate event_dict from tab

    event_dict, df_dict = table2event(
        config, src, icu, table_dict, drops_cols_dict, column_select
    )  # event_dict

    # fail check.
    fail = []
    # icu update using events
    icu_dict = prepare_ICU_class(icu, src)
    for table_name, event_list in event_dict.items():
        for event in event_list:
            if (event.pid in icu_dict.keys()) and (event != []):
                icu_dict[event.pid].events.append(event)  # 这个pid是adm_id
            else:
                fail.append(event)  # check fail ID
    # print('Add all events to icu_dict finish ! ')
    # print("Let's generate data input as numpy file! ")

    '''
    # preparation vocab
    vocab = prep_vocab(icu_dict.values(), column_select)
    if column_select == True:
        code_vocab = pd.DataFrame(columns=['code', 'index'])
        code_vocab['code'] = pd.Series(vocab['code_index'].keys())
        code_vocab['index'] = pd.Series(vocab['code_index'].values())
        code_vocab.to_pickle(os.path.join(
            args.save_path, 'input', args.icu_type, src, 'select', f'code_vocab_{src}.pkl'))

    # time bucektting
    tokenizer = AutoTokenizer.from_pretrained("/home/qluai/ALanZhang/UniHPF/Bio_ClinicalBERT")
    time_delta= []
    for icu in tqdm.tqdm(icu_dict.values()):
        icu.make_sub_token_event(vocab)
        get_sample(icu, args) 
        get_time_bucket(icu, args, src)
        time_delta.extend([event.time_delta for event in icu.events])

    bucket_dict = bucketize(pd.Series(time_delta), quant_num=20)

    for icu in tqdm.tqdm(icu_dict.values()):
        convert2bucket(icu, bucket_dict)

    return icu_dict, vocab
    '''
    return icu_dict


def prep_vocab(icu_list, column_select):
    vocab = dict()
    vocab['token_type_col_content'] = dict({
        '[PAD]': 0, '[CLS]': 1, '[Table]': 2, '[Content]': 3, '[Col]': 4, '[Time]': 5
    })
    # PAD = 0 / CLS = 1 / Table = 2 / Col = 3 / Content = 4 / Time = 5
    vocab['token_class'] = dict({
        '[PAD]': 0, '[CLS]': 1, '[Time]': 2,
    })
    # PAD = 0 / CLS = 1 / Time = 2 / Col_class = 3 ~~~

    vocab_content = list([])
    vocab_column = list([])
    vocab_code = list([])
    for icu in tqdm.tqdm(icu_list):
        icu_content_list = [
            col_content.content
            for event in icu.events
            for col_content in event.col_contents]
        vocab_content.extend(icu_content_list)

        # columns set for col class & Code set
        for event in icu.events:
            vocab_column.extend(list(event.columns))
            if column_select:
                vocab_code.append(event.code_text)
    vocab_content = set(vocab_content)
    vocab_column = set(vocab_column)
    vocab_code = set(vocab_code)

    vocab['content_index'] = dict(zip(
        list(vocab_content), range(14, len(vocab_content) + 14)
    )
    )
    vocab['code_index'] = dict(zip(
        list(vocab_code), range(4, len(vocab_code) + 14)
    )
    )
    vocab['code_index']['[PAD]'] = 0
    vocab['code_index']['[CLS]'] = 1
    vocab['code_index']['[SEP]'] = 2
    vocab['code_index']['[MASK]'] = 3

    # PAD = 0 / CLS = 1 / NaN = 2 / Time = 3~13 / cotent 14~
    for index, col in enumerate(list(vocab_column)):
        vocab['token_class'][col] = 3 + index

    return vocab  # set


def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice
