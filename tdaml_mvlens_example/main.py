import gc
import torch
from meta_embedding import train, train_ctw
from meta_embedding import conf
from hyperopt import fmin, tpe, hp, STATUS_OK
import numpy as np
import os
from meta_embedding import const
import argparse
import random
from hps import models


def get_time_str():
    import time
    return time.strftime('%Y-%m-%d-%H:%M', time.localtime(time.time()))

def hyper_param_analysis(rhos, alphas, path):

    ps = {'alpha': None,
          'amsgrad': False,
          'batch_n_ID': 50,
          'gamma': 1.0,
          'lr': 0.01,
          'p_lr': 0.0001,
          'p_lr_decay': 1.0,
          'rho': None,
          'weight_decay': 1e-8}


    # if os.path.exists(path):
    #     os.remove(path)

    with open(path, 'a') as f:
        f.write('# {}\n'.format(conf.model_type))
        for rho in rhos:
            for alpha in alphas:
                ps['rho'] = rho
                ps['alpha'] = alpha
                # path
                cold_path = "models/mer-trained-rho({})-alpha({}).pth".format(rho, alpha)
                warm_path = 'models/mer-tested-rho({})-alpha({}).pth'.format(rho, alpha)
                # cold
                model = train.new_model(pre_train_path)
                model = train_ctw.meta_train(model, ps, const.train_n_epoch)
                torch.save(model.state_dict(), cold_path)
                cold_loss, cold_auc = train.test(cold_path, warm=False)
                # warm
                ours_warm = train.meta_test(cold_path, 'ours', False)
                warm_losses, warm_aucs = zip(*ours_warm['ours'])
                losses = ',' + ','.join(map(str, warm_losses))
                aucs = ',' + ','.join(map(str, warm_aucs))
                torch.save(model.state_dict(), warm_path)

                msg = ','.join([str(rho), str(alpha), str(cold_loss), str(cold_auc)]) + losses + aucs + '\n'
                print(msg)
                f.write(msg)
                # del model
                # gc.collect()


def build_param_list():
    xs = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ys = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    return xs, ys

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def pre_train(pre_train_path):
    model = train.pre_train()
    loss, auc = train.test(model)
    with open('logs/log.txt', 'a') as f:
        f.write('# {}\n'.format(conf.model_type))
        f.write('base_loss={};\n'.format(loss))
        f.write('base_auc={};\n'.format(auc))
    torch.save(model.state_dict(), pre_train_path)

def me_train(model, me_path):
    me_model = train.meta_train(model)
    torch.save(me_model.state_dict(), me_path)
    loss, auc = train.test(me_model, warm=False)
    me_warm = train.meta_test(me_path, model_name='me', with_normal=True)
    with open('logs/log.txt', 'a') as f:
        f.write('me_loss={};\n'.format(loss))
        f.write('me_auc={};\n'.format(auc))
        f.write('me_warm={};\n'.format(me_warm))

def ours_train(ours_path):
    params = models[conf.model_type]
    fmin(best_auc_func, params, algo=tpe.suggest, max_evals=max_evals, show_progressbar=False)
    loss, auc = train.test(ours_path, warm=False)
    ours_warm = train.meta_test(ours_path, model_name='ours', with_normal=False)
    with open('logs/log.txt', 'a') as f:
        f.write('ours_loss={};\n'.format(loss))
        f.write('ours_auc={};\n'.format(auc))
        f.write('ours_warm={};\n'.format(ours_warm))
    global best_params
    with open('logs/best.txt', 'a') as f:
        f.write('# {}\n'.format(conf.model_type))
        f.write('best_{}={};\n'.format(conf.model_type, best_params))


def best_auc_func(params):
    model = train.new_model(pre_train_path)
    train_ctw.meta_train(model, params)
    loss, auc = train.test(model, warm=False)
    global best_auc
    global best_params
    if auc > best_auc:
        torch.save(model.state_dict(), ours_path)
        best_auc = auc
        best_params = params
        print('best:{}'.format(best_auc))
    return {'loss': -auc, 'status': STATUS_OK}


def save_tables():
    for model_type in ['FM', 'DNN', 'WD', 'DeepFM']:
    # for model_type in ['DNN', 'WD', 'DeepFM']:
        conf.model_type = model_type

        pre_train_path1 = "models/pre-trained-{}-{}.pth".format(model_type, conf.cvr_flag)
        me_path1 = "models/me-{}-{}.pth".format(model_type, conf.cvr_flag)
        ours_path1 = "models/ours-{}-{}.pth".format(model_type, conf.cvr_flag)

        conf.table_path = 'logs/table_base_cold_{}.csv'.format(model_type)
        train.test(pre_train_path1, warm=False, save_table=True)

        conf.table_path = 'logs/table_me_cold_{}.csv'.format(model_type)
        train.test(me_path1, warm=False, save_table=True)

        conf.table_path = 'logs/table_ours_cold_{}.csv'.format(model_type)
        train.test(ours_path1, warm=False, save_table=True)

        conf.table_path = 'logs/table_base_warm_{}.csv'.format(model_type)
        train.meta_test(pre_train_path1, with_normal=False, save_table=True, base=True)

        conf.table_path = 'logs/table_me_warm_{}.csv'.format(model_type)
        train.meta_test(me_path1, with_normal=False, save_table=True)

        conf.table_path = 'logs/table_ours_warm_{}.csv'.format(model_type)
        train.meta_test(ours_path1, with_normal=False, save_table=True)

        print('{}-------------saved!!!!!!!!!!!!!!!!!!!!!'.format(model_type))



if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='MER.')
    # parser.add_argument('--device', type=str)
    # parser.add_argument('--i1', type=int)
    # parser.add_argument('--i2', type=int)
    # parser.add_argument('--savepath', type=str)
    # parser.add_argument('--cvrflag', type=int)
    # args = parser.parse_args()
    #
    # conf.device = torch.device(args.device)
    # conf.cvr_flag = args.cvrflag
    # i1, i2 = args.i1, args.i2
    # save_path = args.savepath

    conf.device = torch.device('cuda:0')
    conf.cvr_flag = 0
    i1, i2 = 0, 1
    save_path_ = 'models/models.txt'
    conf.model_type = 'DeepFM' # ['FM', 'DNN', 'WD', 'DeepFM']
    max_evals = 1
    setup_seed(0)

    best_auc = 0.0
    bset_params = 0.0

    pre_train_path = "models/pre-trained-{}-{}.pth".format(conf.model_type, conf.cvr_flag)
    me_path = "models/me-{}-{}.pth".format(conf.model_type, conf.cvr_flag)
    ours_path = "models/ours-{}-{}.pth".format(conf.model_type, conf.cvr_flag)
    # save_path_ = 'models/models.txt'

    pre_train(pre_train_path)
    me_train(pre_train_path, me_path)
    ours_train(ours_path)
