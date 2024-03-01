from collections import OrderedDict

from preprocess.parse_csv import EHRParser
import os
'''
code_map将每个code映射为一个索引
admission_codes_encoded 将admission_codes中的codes变成映射的索引
'''
def encode_code(patient_admission, admission_codes, admDxMap, admDxMap_css, admDxMap_css_cat1, seqs_ccs, seqs_ccs_cat1, seqs):
    code_map = OrderedDict()
    dict_ccs = {}
    dict_ccs_cat1 = {}
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            codes = admission_codes[admission[EHRParser.adm_id_col]]
            codes_ccs = admDxMap_css[admission[EHRParser.adm_id_col]]
            codes_ccs_cat1 = admDxMap_css_cat1[admission[EHRParser.adm_id_col]]
            for code in codes:
                if code not in code_map:
                    code_map[code] = len(code_map)
            for code_ccs in codes_ccs:
                if code_ccs not in dict_ccs:
                    dict_ccs[code_ccs] = len(dict_ccs)

            for code_ccs_cat1 in codes_ccs_cat1:
                if code_ccs_cat1 not in dict_ccs_cat1:
                    dict_ccs_cat1[code_ccs_cat1] = len(dict_ccs_cat1)
    admission_codes_encoded = {
        admission_id: list(set(code_map[code] for code in codes))
        for admission_id, codes in admission_codes.items()
    }
    vocab_set = {}
    for i, seq in enumerate(seqs):
        for visit in seq:
            for code in visit:
                if code in vocab_set:  # vocab_set code出现过几次
                    vocab_set[code] += 1
                else:
                    vocab_set[code] = 1
    sorted_vocab = {k: v for k, v in sorted(vocab_set.items(), key=lambda item: item[1],
                                            reverse=True)}  # 按照每个item的items[0] reverse=True: 逆序排列。默认从小到大，逆序后从大到小。
    outfd = open(os.path.join('data/mimic4/parsed', 'vocab.txt'), 'w')
    for k, v in sorted_vocab.items():
        outfd.write(k + '\n')
    outfd.close()

    print('codes num: %d' % len(code_map))
    return admission_codes_encoded, code_map, dict_ccs, dict_ccs_cat1, vocab_set
