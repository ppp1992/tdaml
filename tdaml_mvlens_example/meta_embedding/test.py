import gc
import numpy as np
import torch.nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
from meta_embedding.const import *
from meta_embedding.dataset import BatchLoader
from meta_embedding.meta_model import MetaModel


def test(model, warm=True, save_table=False):
    model = new_model(model) if isinstance(model, str) else model
    model.eval()
    y, _ = model(t_data, warm)
    y_pred = y.flatten()
    y_true = torch.from_numpy(t_label).to(conf.device, dtype=torch.float)

    loss = criterion(y_pred, y_true).item()

    y_true_ = y_true.cpu().detach().numpy()
    y_pred_ = y_pred.cpu().detach().numpy()
    auc = roc_auc_score(y_true_, y_pred_)
    # print('\n\t- testing loss:', loss)
    # print('\t- testing auc:', auc)

    # save the result in file to count per-category
    if save_table:
        df: pd.DataFrame = t_data.copy()
        df['y_true'] = y_true_
        df['y_pred'] = y_pred_
        df.to_csv(conf.table_path, index=False)
        print('save one table!')

    return loss, auc

fs_loaders_mt = None
def meta_test(model_path: str, model_name=None, with_normal=True, save_table=False, base=False):
    print('[meta test]')

    paths = ['data/test_oneshot_a.pkl',
             'data/test_oneshot_b.pkl',
             'data/test_oneshot_c.pkl']
    if conf.cvr_flag != 0:
        paths = ['data/data2/test_1_{}.pkl'.format(i) for i in range(6)]

    task_size = 20
    batch_n_ID = 25
    batch_size = task_size * batch_n_ID
    # i = 1
    global fs_loaders_mt
    if fs_loaders_mt is None:
        fs_loaders = pool.map(lambda p: BatchLoader(p, padding_conf, batch_size), paths, 1)
    else:
        fs_loaders = fs_loaders_mt
    fs_loaders_mt = fs_loaders
    test_n_ID = len(fs_loaders[0].data[id_col].drop_duplicates())

    batch_n = int(np.ceil(test_n_ID / batch_n_ID))

    n_epoch = test_n_epoch

    result = {}

    if with_normal or base:
        print('meta test --- normal')
        result['base'] = []

        model = new_model(model_path)
        optimizer = torch.optim.Adam(params=model.embs[0].parameters(), lr=1e-3)

        for epoch_i_ in range(n_epoch):
            epoch_i = epoch_i_ % 3
            model.train()

            for i in tqdm(range(batch_n), 'epoch {} '.format(epoch_i_)):
                data_a, label_a = fs_loaders[epoch_i].load(i, cols, target_col=target_col)
                y, _ = model(data_a)
                y_pred = y.flatten()

                y_true = torch.from_numpy(label_a).float().to(conf.device)
                loss = criterion(y_pred, y_true)

                # back prop.
                model.zero_grad()
                loss.backward()
                optimizer.step()
            result['base'].append(test(model))

        if save_table:
            test(model, save_table=True)
        if base:
            return result

    print('===================================')

    print('meta test --- maml')
    model = new_model(model_path)
    optimizer = torch.optim.Adam(params=model.embs[0].parameters(), lr=1e-3)

    if model_name == None:
        model_name = model_path
    result[model_name] = []

    for epoch_i_ in range(n_epoch):
        epoch_i = epoch_i_ % 3
        model.train()
        for i in tqdm(range(batch_n), 'epoch {} '.format(epoch_i_)):
            data_a, label_a = fs_loaders[epoch_i].load(i, cols, target_col=target_col)

            if epoch_i_ == 0:
                _, meta_id_emb = model(data_a, warm=False, emb_only=True)
                for k in range(batch_n_ID):
                    # find a task(ad)
                    start = k * task_size
                    end = min((k + 1) * task_size, len(meta_id_emb))
                    if start >= len(meta_id_emb):
                        break
                    id_val = data_a[id_col].values[start]
                    id_emb = meta_id_emb[start: end].mean(dim=0)
                    model.set_id_emb(id_val, id_emb)

            y, _ = model(data_a)

            y_pred = y.flatten()
            y_true = torch.from_numpy(label_a).float().to(conf.device)
            loss = criterion(y_pred, y_true)

            model.zero_grad()
            loss.backward()
            optimizer.step()

        result[model_name].append(test(model))
    if save_table:
        test(model, save_table=True)
    return result