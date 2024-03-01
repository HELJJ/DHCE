import os
import pdb
import random
import time
import pickle
import torch
import torch.nn.functional as F
from shutil import copyfile
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from models.modeling import BertConfig
from preprocess.function_helpers import build_tree_with_padding, get_rootCode
from models.model_events_sum  import Model
from utils import load_adj, EHRDataset, format_time, MultiStepLRScheduler
from metrics import evaluate_codes, evaluate_hf
from transformers import logging
#from torch.utils.tensorboard import SummaryWriter
logging.set_verbosity_warning()
logging.set_verbosity_error()

def historical_hot(code_x, code_num, lens):
    '''
    :param result: shape:(code_x_num,code_num)
    :param code_x: shape:(train_pid_nums,max_admission_num,code_num)
    :param code_num:
    :param lens:每个pid患者的历史诊断次数
    :return: 返回每个pid患者的前一次诊断
    '''
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
if __name__ == '__main__':
    seed = 6669
    dataset = 'mimic3'  # 'mimic3' or 'eicu'
    task = 'm'  # 'm' or 'h'
    use_cuda = True

    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print("MIMIC:",dataset)
    print("device:", device)
    code_size = 48 #diagnoses embedding和neighbor embedding
    graph_size = 32#unrelated disease embedding
    hidden_size = 512  # rnn hidden size #MIMIC-III：256；MIMIC-IV：350
    t_attention_size = 32 #32
    t_output_size = hidden_size
    batch_size = 32
    epochs = 200

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #添加外部知识的相关代码
    # training data files
    seqs_file = os.path.join('/home/qlunlp/workplace1/qluai/Zhang/Chet_integrate/data/mimic3/standard/train', 'inputs.seqs')
    print(seqs_file)
    # dictionary files
    dict_file = os.path.join('/home/qlunlp/workplace1/qluai/Zhang/Chet_integrate/data/mimic3', 'inputs.dict')
    tree_dir = '/home/qlunlp/workplace1/qluai/Zhang/Chet_integrate/data/mimic3'
    class_dict_file = os.path.join('/home/qlunlp/workplace1/qluai/Zhang/Chet_integrate/data/mimic3/parsed/ccs_single_level.dict')
    visit_class_dict_file = os.path.join('/home/qlunlp/workplace1/qluai/Zhang/Chet_integrate/data/mimic3/parsed/ccs_cat1.dict')
    code2desc_file = os.path.join('/home/qlunlp/workplace1/qluai/Zhang/Chet_integrate/data/mimic3/code2desc.dict')
    leaves_list = []
    ancestors_list = []
    masks_list = []
    for i in range(5, 0, -1):
        leaves, ancestors, masks = build_tree_with_padding(os.path.join(tree_dir, 'level' + str(i) + '.pk'))
        leaves_list.extend(leaves)
        ancestors_list.extend(ancestors)
        masks_list.extend(masks)
    leaves_list = torch.tensor(leaves_list).long().to(device)
    ancestors_list = torch.tensor(ancestors_list).long().to(device)
    masks_list = torch.tensor(masks_list).float().to(device)
    # load configure file
    output_dir = os.path.join('/home/ubuntu/Zhang/chet_know_diaocan_2.23/data/', 'mimic')
    config_json = 'models/config.json'
    # config_json = 'KEMCE/model/config.json'
    #copyfile(config_json, os.path.join(output_dir, 'config.json'))
    config = BertConfig.from_json_file(config_json)
    config.leaves_list = leaves_list
    config.ancestors_list = ancestors_list
    config.masks_list = masks_list
    vocab = pickle.load(open(dict_file, 'rb'))
    config.code_size = len(vocab)
    num_tree_nodes = get_rootCode(os.path.join(tree_dir, 'level2.pk')) + 1
    config.num_tree_nodes = num_tree_nodes
    class_vocab = pickle.load(open(class_dict_file, 'rb'))
    config.num_ccs_classes = len(class_vocab)
    visit_class_vocab = pickle.load(open(visit_class_dict_file, 'rb'))
    config.num_visit_classes = len(visit_class_vocab)
    config.device = device
    #添加外部知识的相关代码



    dataset_path = os.path.join('data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    code_adj = load_adj(dataset_path, device=device)#加载邻接矩阵转为tensor
    code_num = len(code_adj)
    print('loading train data ...')
    train_data = EHRDataset(code_adj, train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDataset(code_adj, valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('loading test data ...')
    test_data = EHRDataset(code_adj, test_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    test_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)
    task_conf = {
        'm': {
            'dropout': 0.45,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.01,
                'milestones': [20, 30],
                'lrs': [1e-3, 1e-5]
            }
        },
        'h': {
            'dropout': 0.0,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }
    }
    output_size = task_conf[task]['output_size']
    activation = torch.nn.Sigmoid()
    loss_fn = torch.nn.BCELoss()
    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = task_conf[task]['dropout']

    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    model = Model(config = config, code_num=code_num, code_size=code_size,
                  adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
                  t_output_size=t_output_size,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
                                     task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    best_auc = 0
    best_f1 = 0
    best_recall_10=0
    best_recall_20=0
    # writer1 = SummaryWriter('/home/ubuntu/Zhang/integrate/runs/trainloss')
    # writer2 = SummaryWriter('/home/ubuntu/Zhang/integrate/runs/validloss')
    # writer3 = SummaryWriter('/home/ubuntu/Zhang/integrate/runs/f1')
    # 创建训练和测试的SummaryWriter
    #train_writer = SummaryWriter('./runs_4.25/mimic3_know/train/')
    #valid_writer = SummaryWriter('./runs_4.25/mimic3_know/valid/')
    def f1(y_true_hot, y_pred, metrics='weighted'):
        result = np.zeros_like(y_true_hot)
        for i in range(len(result)):
            true_number = np.sum(y_true_hot[i] == 1)
            result[i][y_pred[i][:true_number]] = 1
        return f1_score(y_true=y_true_hot, y_pred=result, average=metrics, zero_division=0)
    # 定义一个函数，用于记录loss、学习率和准确率
    def log_metrics(writer, loss, learning_rate, accuracy, step):
        writer.add_scalar("loss", loss, step)
        writer.add_scalar("learning_rate", learning_rate, step)
        writer.add_scalar("accuracy", accuracy, step)
    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        scheduler.step()
        train_preds = []
        train_labels = train_data.label()
        for step in range(len(train_data)):
            optimizer.zero_grad()
            code_x, visit_lens, divided, y, neighbors, events, know_code_inputs, know_code_mask,  diagnosis_list = train_data[step]
            output = model(code_x, divided, neighbors, visit_lens, events, know_code_inputs, know_code_mask,  diagnosis_list).squeeze()
            output_1 = model(code_x, divided, neighbors, visit_lens, events, know_code_inputs, know_code_mask,  diagnosis_list).squeeze()
            #pdb.set_trace()
            train_pred = torch.argsort(output, dim=-1, descending=True)
            train_preds.append(train_pred)
            #loss = loss_fn(output, y)

            ce_loss = 0.5 * (loss_fn(output, y) + loss_fn(output_1, y))
            #loss = loss.detach_().requires_grad_(True)
            kl_loss = compute_kl_loss(output, output_1)
            #carefully choose hyper-parameters
            loss = ce_loss + 0.2 * kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)
            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                  % (step + 1, steps, remaining_time, total_loss / total_num), end='')
        train_preds = torch.vstack(train_preds).detach().cpu().numpy()
        train_f1_score = f1(train_labels, train_preds)
        # 记录loss、学习率和准确率
        log_metrics(train_writer, total_loss / total_num, optimizer.param_groups[0]['lr'], train_f1_score, epoch)
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))
        if task == 'h':
            valid_loss, f1_score, auc= evaluate_fn(model, valid_data, loss_fn, output_size, test_historical)
            if best_auc<auc:
                best_auc=auc
            if f1_score>best_f1:
                best_f1=f1_score
            #torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))
        if task == 'm':
            valid_loss, evalute_f1_score,recall = evaluate_fn(model, valid_data, loss_fn, output_size, test_historical)
            log_metrics(valid_writer, valid_loss, optimizer.param_groups[0]['lr'], evalute_f1_score, epoch)
            if best_recall_10<recall[0]:
                best_recall_10=recall[0]
            if best_recall_20<recall[1]:
                best_recall_20=recall[1]
            if evalute_f1_score>best_f1:
                best_f1=evalute_f1_score

            #torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))
                # torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))


    if task=='h':
        print("best_auc=%.4f, best_f1=%.4f"%(best_auc, best_f1))
    if task == 'm':
        print("best_recall_10=%.4f,best_recall_20=%.4f,best_f1=%.4f" % (best_recall_10, best_recall_20, best_f1))




