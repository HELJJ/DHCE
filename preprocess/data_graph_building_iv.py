import pickle
import collections
import argparse
import os
import numpy as np
from icd9cms.icd9 import search


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    vocab['PAD'] = len(vocab)
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = len(vocab)
    return vocab


def tree_building(train_seqs, valid_seqs, test_seqs, types_file, graph_file, out_file):
    #seqs_file:inputs_all.seqs
    #types_file:vocab.txt
    #graph_file:ccs_multi_dx_tool_2015.csv
    #out_file:/output_path/mimic
    print('Read Saved data dictionary')
    np.load.__defaults__ = (None, True, True, 'ASCII')
    types = load_vocab(types_file)
    train_seqs = np.load(train_seqs)
    valid_seqs = np.load(valid_seqs)
    test_seqs = np.load(test_seqs)
    # know_neighbor_code = np.load(os.path.join(self.path, 'neighbors_code.npy'))
    np.load.__defaults__ = (None, False, True, 'ASCII')



    start_set = set(types.keys())
    hit_list = []
    code2desc = collections.OrderedDict()

    cat1count = 0
    cat2count = 0
    cat3count = 0
    cat4count = 0

    graph = open(graph_file, 'r')
    _ = graph.readline()

    # add ancestors to dictionary
    for line in graph:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_L1_' + cat1
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_L2_' + cat2
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_L3_' + cat3
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_L4_' + cat4

        icd9 = 'D_' + icd9

        if icd9 not in types:
            continue
        else:
            hit_list.append(icd9)
            sr = search(icd9[2:]) #sr得到icd-9:short_desc:long_desc
            if sr is not None:
                ds = str(sr).split(':') #
                if ds[2] == 'None':
                    code2desc[icd9] = ds[1]
                else:
                    code2desc[icd9] = ds[2]
            else:
                code2desc[icd9] = '[UNK]'

        if desc1 not in types:
            cat1count += 1
            types[desc1] = len(types)
            desc= tokens[2][1:-1].strip()
            code2desc[desc1] = desc

        if len(cat2) > 0:
            if desc2 not in types:
                cat2count += 1
                types[desc2] = len(types)
                desc = tokens[4][1:-1].strip()
                code2desc[desc2] = desc
        if len(cat3) > 0:
            if desc3 not in types:
                cat3count += 1
                types[desc3] = len(types)
                desc = tokens[6][1:-1].strip()
                code2desc[desc3] = desc
        if len(cat4) > 0:
            if desc4 not in types:
                cat4count += 1
                types[desc4] = len(types)
                desc = tokens[8][1:-1].strip()
                code2desc[desc4] = desc
    graph.close()

    # add root_code
    types['A_ROOT'] = len(types)
    root_code = types['A_ROOT']
    code2desc['A_ROOT'] = '[UNK]'

    miss_set = start_set - set(hit_list)
    miss_set.remove('PAD')  # comment this line for GRAM and KAME, work for KEMCE
    print('missing code: {}'.format(len(miss_set)))
    print("11", cat1count, cat2count, cat3count, cat4count)

    five_map = {}
    four_map = {}
    three_map = {}
    two_map = {}
    one_map = dict([(types[icd], [types[icd], root_code]) for icd in miss_set])

    graph = open(graph_file, 'r')
    graph.readline()

    for line in graph:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_L1_' + cat1
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_L2_' + cat2
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_L3_' + cat3
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_L4_' + cat4

        icd9 = 'D_' + icd9

        if icd9 not in types:
            continue

        icd_code = types[icd9]

        if len(cat4) > 0:
            code4 = types[desc4]
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            five_map[icd_code] = [icd_code, root_code, code1, code2, code3, code4]
        elif len(cat3) > 0:
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            four_map[icd_code] = [icd_code, root_code, code1, code2, code3]

        elif len(cat2) > 0:
            code2 = types[desc2]
            code1 = types[desc1]
            three_map[icd_code] = [icd_code, root_code, code1, code2]

        else:
            code1 = types[desc1]
            two_map[icd_code] = [icd_code, root_code, code1]

    # Now we re-map the integers to all medical leaf codes.
    new_five_map = collections.OrderedDict()
    new_four_map = collections.OrderedDict()
    new_three_map = collections.OrderedDict()
    new_two_map = collections.OrderedDict()
    new_one_map = collections.OrderedDict()
    new_types = collections.OrderedDict()
    new_code2desc = collections.OrderedDict()
    types_reverse = dict([(v, k) for k, v in types.items()])

    code_count = 0
    new_types['PAD'] = code_count  # comment this line for GRAM and KAME, work for KEMCE
    code_count += 1  # comment this line for GRAM and KAME, work for KEMCE
    new_code2desc['PAD'] = '[UNK]'
    for icdCode, ancestors in five_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_five_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        new_code2desc[types_reverse[icdCode]] = code2desc[types_reverse[icdCode]]

    for icdCode, ancestors in four_map.items():
        code = types_reverse[icdCode]
        # print(code)
        new_types[code] = code_count
        new_four_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        new_code2desc[code] = code2desc[code]

    for icdCode, ancestors in three_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_three_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        new_code2desc[types_reverse[icdCode]] = code2desc[types_reverse[icdCode]]

    for icdCode, ancestors in two_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_two_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        new_code2desc[types_reverse[icdCode]] = code2desc[types_reverse[icdCode]]

    for icdCode, ancestors in one_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_one_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        new_code2desc[types_reverse[icdCode]] = code2desc[types_reverse[icdCode]]

    for code in code2desc.keys():
        if code not in new_code2desc:
            new_code2desc[code] = code2desc[code]

    train_new_seqs = []
    for patient in train_seqs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in visit:
                new_visit.append(new_types[code])
            new_patient.append(new_visit)
        train_new_seqs.append(new_patient)

    valid_new_seqs = []
    for patient in valid_seqs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in visit:
                new_visit.append(new_types[code])
            new_patient.append(new_visit)
        valid_new_seqs.append(new_patient)

    test_new_seqs = []
    for patient in test_seqs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in visit:
                new_visit.append(new_types[code])
            new_patient.append(new_visit)
        test_new_seqs.append(new_patient)
    print("saving....")
    pickle.dump(new_five_map, open(os.path.join(out_file, 'level5.pk'), 'wb'), -1)
    pickle.dump(new_four_map, open(os.path.join(out_file, 'level4.pk'), 'wb'), -1)
    pickle.dump(new_three_map, open(os.path.join(out_file, 'level3.pk'), 'wb'), -1)
    pickle.dump(new_two_map, open(os.path.join(out_file, 'level2.pk'), 'wb'), -1)
    pickle.dump(new_one_map, open(os.path.join(out_file, 'level1.pk'), 'wb'), -1)
    pickle.dump(new_types, open(os.path.join(out_file, 'inputs.dict'), 'wb'), -1)
    pickle.dump(train_new_seqs, open(os.path.join('../data/mimic4/standard/train', 'inputs.seqs'), 'wb'), -1)
    pickle.dump(valid_new_seqs, open(os.path.join('../data/mimic4/standard/valid', 'inputs.seqs'), 'wb'), -1)
    pickle.dump(test_new_seqs, open(os.path.join('../data/mimic4/standard/test', 'inputs.seqs'), 'wb'), -1)
    pickle.dump(new_code2desc, open(os.path.join(out_file, 'code2desc.dict'), 'wb'), -1)
    print("ending...")
    #print(len(new_types), len(new_seqs), len(new_code2desc), ancestors[1])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output",
                        type=str,
                        #required=True,
                        default='/home/ubuntu/Zhang/integrate/data/mimic4',
                        help="The output directory where the processed files will be written.")
    parser.add_argument("--train_seqs",
                        type=str,
                        #required=True,
                        default='/home/ubuntu/Zhang/integrate/data/mimic4/standard/train/know_code.npy',
                        help="The path of train seqs file.")
    parser.add_argument("--valid_seqs",
                        type=str,
                        # required=True,
                        default='/home/ubuntu/Zhang/integrate/data/mimic4/standard/valid/know_code.npy',
                        help="The path of valid seqs file.")
    parser.add_argument("--test_seqs",
                        type=str,
                        # required=True,
                        default='/home/ubuntu/Zhang/integrate/data/mimic4/standard/test/know_code.npy',
                        help="The path of test seqs file.")
    parser.add_argument("--vocab",
                        type=str,
                        #required=True,
                        default='/home/ubuntu/Zhang/integrate/data/mimic4/parsed/vocab.txt',
                        help="The path of vocabulary file.")
    parser.add_argument("--multi_level",
                        # default=None,
                        type=str,
                        #required=True,
                        default='/home/ubuntu/Zhang/integrate/ccs/ccs_multi_dx_tool_2015.csv',
                        help="The path of CCS multi-level of diagnoses.")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    tree_building(args.train_seqs, args.valid_seqs, args.test_seqs, args.vocab, args.multi_level, args.output) #这个地方的args.seqs是训练、测试、验证集中的seqs，


