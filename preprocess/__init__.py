import os
import numpy as np
import pickle

def save_sparse(path, x):
    idx = np.where(x > 0)
    values = x[idx]
    np.savez(path, idx=idx, values=values, shape=x.shape)


def load_sparse(path):
    data = np.load(path)
    idx, values = data['idx'], data['values']
    mat = np.zeros(data['shape'], dtype=values.dtype)
    mat[tuple(idx)] = values
    return mat


def save_data(path, code_x, visit_lens, codes_y, divided, neighbors, events, know_code, neighbors_code):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'code_y'), codes_y)
    #np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    save_sparse(os.path.join(path, 'divided'), divided)
    save_sparse(os.path.join(path, 'neighbors'), neighbors)
    # np.savez(os.path.join(path, 'events'), events=events)
    # 将子列表转换为NumPy数组并堆叠在一起形成2D数组
    #know_code_array = np.vstack([np.array(x) for x in know_code])
    know_code_array_object = np.array(know_code, dtype=object)
    np.save(os.path.join(path, 'know_code.npy'), know_code_array_object)
    neighbors_code_array_object = np.array(neighbors_code, dtype=object)
    np.save(os.path.join(path, 'neighbors_code.npy'), neighbors_code_array_object)
    filePath = os.path.join(path, 'event.pkl')
    with open(filePath,'wb') as fo:
        pickle.dump(events, fo)
        fo.close()
