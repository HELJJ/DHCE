import os

import torch
import numpy as np
import pdb
from preprocess import load_sparse
import pickle

def load_adj(path, device=torch.device('cpu')):
    filename = os.path.join(path, 'code_adj.npz')
    adj = torch.from_numpy(load_sparse(filename)).to(device=device, dtype=torch.float32)
    return adj


class EHRDataset:
    def __init__(self, code_adj, data_path, label='m', batch_size=32, shuffle=True, device=torch.device('cpu')):
        super().__init__()
        self.code_adj = code_adj
        self.path = data_path
        self.code_x, self.visit_lens, self.y, self.divided, self.neighbors, self.event_dict, self.know_code= self._load(label)

        self._size = self.code_x.shape[0]
        self.idx = np.arange(self._size) #self._size train患者 pid 数量
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def _load(self, label):
        code_x = load_sparse(os.path.join(self.path, 'code_x.npz'))
        visit_lens = np.load(os.path.join(self.path, 'visit_lens.npz'))['lens']
        if label == 'm':
            y = load_sparse(os.path.join(self.path, 'code_y.npz'))
        elif label == 'h':
            y = np.load(os.path.join(self.path, 'hf_y.npz'))['hf_y']
        else:
            raise KeyError('Unsupported label type')
        divided = load_sparse(os.path.join(self.path, 'divided.npz'))
        neighbors = load_sparse(os.path.join(self.path, 'neighbors.npz'))
        event_dict = pickle.load(open(os.path.join(self.path, 'event.pkl'), 'rb'))
        #
        np.load.__defaults__ = (None, True, True, 'ASCII')
        know_code = pickle.load(open(os.path.join(self.path, 'inputs.seqs'), 'rb'))
        # know_neighbor_code = np.load(os.path.join(self.path, 'neighbors_code.npy'))
        np.load.__defaults__ = (None, False, True, 'ASCII')

        return code_x, visit_lens, y, divided, neighbors, event_dict, know_code

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def size(self):
        return self._size

    def label(self):
        return self.y

    def __len__(self):
        len_ = self._size // self.batch_size
        return len_ if self._size % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index):
        device = self.device
        start = index * self.batch_size
        end = start + self.batch_size
        slices = self.idx[start:end]
        code_adj_numpy = self.code_adj.cpu().numpy()
        code_x = torch.from_numpy(self.code_x[slices]).to(device)
        visit_lens = torch.from_numpy(self.visit_lens[slices]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(self.y[slices]).to(device=device, dtype=torch.float32)
        divided = torch.from_numpy(self.divided[slices]).to(device)
        neighbors = torch.from_numpy(self.neighbors[slices]).to(device)
        clinical_events = self.event_dict[start:end]#[patient[visits[events]]]]

        max_visit_len = 0
        max_batch_size = 0
        max_events_len = 0
        for batch_size in clinical_events:
            if len(batch_size) > max_batch_size:
                max_batch_size = len(batch_size)
            for visits in batch_size:
                if len(visits) > max_visit_len:
                    max_visit_len = len(visits)
                for events in visits:
                    if len(events) > max_events_len:
                        max_events_len = len(events)
        for batch_size_i in clinical_events:
            if len(batch_size_i) < max_batch_size:
                for new_batchs_size_num in range(max_batch_size - len(batch_size_i)):
                    new_batch_size = [[0 for j in range(max_events_len)] for i in range(max_visit_len)]
                    batch_size_i.append(new_batch_size)
            for visits_i in batch_size_i:
                if len(visits_i) < max_visit_len:
                    for h in range(max_visit_len - len(visits_i)):
                        new_list_visits = [0] * max_events_len
                        visits_i.append(new_list_visits)
                for events_i in visits_i:
                    if len(events_i) < max_events_len:
                        new_list_events = [0] * (max_events_len - len(events_i))
                        events_i += new_list_events

        #know_code_array = np.array([np.array(xi) for xi in self.know_code])
        know_code = [self.know_code[i] for i in slices]
        #know_code = know_code_array[slices]
        seq_lens = [len(x) for x in know_code]  # batch里面所有患者的就诊次数
        max_seq_len = max(seq_lens)  # 求batch里面最大就诊长度
        visit_mask_pad = []
        for l in seq_lens:
            mask = np.zeros(max_seq_len, dtype=np.float32)
            mask[:l] = 1
            visit_mask_pad.append(mask)
        visit_mask = np.stack(visit_mask_pad)
        max_visit_len = 0
        for x in know_code:
            for visit in x:
                if max_visit_len < len(visit):
                    max_visit_len = len(visit)

        inputs_pad = []
        for x in know_code:
            seq_len = len(x)
            visit_pad = np.zeros((max_seq_len - seq_len, max_visit_len), dtype=np.float32)
            visit_codes = []
            for visit in x:
                visit_len = len(visit)

                code_pad = visit + [0] * (max_visit_len - visit_len)
                visit_codes.append(code_pad)
            seq_pad = np.concatenate([np.stack(visit_codes), visit_pad], axis=0)
            inputs_pad.append(seq_pad)
        inputs = np.stack(inputs_pad)
        code_mask = np.array(inputs > 0, dtype=np.float32)
        # 在adj中提取诊断疾病权重矩阵
        extract_diagnosis_adj = []
        patient_diagnosis_list = []

        for patient, seq_len_i in zip(code_x, visit_lens):  # 遍历每一个患者
            visit_weight = []  # 每一次就诊的权重矩阵
            visit_lists = []  # 每一次就诊诊断疾病列表
            for t, (visit, seq_len_i) in enumerate(zip(patient, range(seq_len_i))):  # 遍历患者每一次就诊 得到每一次就诊的权重矩阵
                visit = visit.cpu().numpy()
                visit_list = visit.nonzero()[0].tolist()
                visit_lists.append(visit_list)
                '''
                # 现在adj中取出列来
                extract_colum = code_adj_numpy[:, visit_list]
                # 在adj中取出行来
                extract_row_colum = extract_colum[visit_list, :]
                # 然后补齐每一行每一列
                # 先构造一个矩阵 作为补齐行的矩阵 这个补齐就是让batch内所有患者每一次的诊断code长度都一致

                extract_row_colum_pad = np.zeros((max_visit_len - extract_row_colum.shape[0], max_visit_len),
                                                 dtype=np.float32)

                # 遍历每一行将列补齐
                new_extract_row_colum = []
                for i, line in enumerate(extract_row_colum):
                    line_len = len(line)  # 数组长度 一维数组
                    line = line.tolist() + [0] * (max_visit_len - line_len)  # 对这个一维数组的列补齐
                    new_extract_row_colum.append(line)
                # 然后把行补齐
                seq_pad = np.concatenate([np.stack(new_extract_row_colum), extract_row_colum_pad], axis=0)
                seq_pad = seq_pad.astype(np.float32)
                visit_weight.append(seq_pad)
            '''
            #extract_diagnosis_adj.append(np.array(visit_weight))
            patient_diagnosis_list.append(visit_lists)

        #extract_diagnosis_adj = np.array(extract_diagnosis_adj)

        # 将列表转换为PyTorch张量
        patient_diagnosis_list_tensor = torch.zeros(len(patient_diagnosis_list), max(map(len, patient_diagnosis_list)),
                                max(map(len, [item for sublist in patient_diagnosis_list for item in sublist]))).long()

        for i, sublist in enumerate(patient_diagnosis_list):
            for j, item in enumerate(sublist):
                for k, val in enumerate(item):
                    patient_diagnosis_list_tensor[i, j, k] = val
        self.patient_diagnosis_list_to_tensor = patient_diagnosis_list_tensor
        clinical_events = torch.tensor(clinical_events).to(device)
        return code_x, visit_lens, divided, y, neighbors, clinical_events, \
               torch.tensor(inputs).long().to(device), torch.tensor(code_mask).long().to(device), self.patient_diagnosis_list_to_tensor.to(device)

class MultiStepLRScheduler:
    def __init__(self, optimizer, epochs, init_lr, milestones, lrs):
        self.optimizer = optimizer
        self.epochs = epochs
        self.init_lr = init_lr
        self.lrs = self._generate_lr(milestones, lrs)
        self.current_epoch = 0

    def _generate_lr(self, milestones, lrs):
        milestones = [1] + milestones + [self.epochs + 1]
        lrs = [self.init_lr] + lrs
        lr_grouped = np.concatenate([np.ones((milestones[i + 1] - milestones[i], )) * lrs[i]
                                     for i in range(len(milestones) - 1)])
        return lr_grouped

    def step(self):
        lr = self.lrs[self.current_epoch]
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.current_epoch += 1

    def reset(self):
        self.current_epoch = 0


def format_time(seconds):
    if seconds <= 60:
        time_str = '%.1fs' % seconds
    elif seconds <= 3600:
        time_str = '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        time_str = '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    return time_str
