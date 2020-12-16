import torch.nn
from torch.optim.lr_scheduler import *
from torch.autograd import grad
# from tensorboardX import SummaryWriter
from meta_embedding.projection import project_onto_chi_square_ball
from meta_embedding.test import *
from numba import jit


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 1e-3 * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

fs_loaders_m = None
def meta_train(model: MetaModel, params, n_epoch=3):
    print('meta train')

    model = new_model(model) if isinstance(model, str) else model
    paths = ['data/train_oneshot_a.pkl',
             'data/train_oneshot_b.pkl',
             'data/train_oneshot_c.pkl']
    if conf.cvr_flag != 0:
        paths = ['data/data2/train_{}.pkl'.format(i) for i in range(5)]

    optimizer = torch.optim.Adam(params=model.generator.parameters(),
                                 lr=params['lr'],
                                 weight_decay=params['weight_decay'],
                                 amsgrad=params['amsgrad'])
    if params['gamma'] < 1.0:
        scheduler = ReduceLROnPlateau(optimizer, factor=params['gamma'], patience=3, verbose=True)
    # scheduler = StepLR(optimizer, step_size=5, gamma=params['gamma'])
    # scheduler = MultiStepLR(optimizer, milestones=[4, 10, 20, 30, 40], gamma=0.5)
    # scheduler = CosineAnnealingLr(optimizer, T_max=5, eta_min=0)

    task_size = 20
    batch_n_ID = int(params['batch_n_ID'])
    global fs_loaders_m

    if fs_loaders_m is None:
        fs_loaders =  [BatchLoader(p, padding_conf, task_size) for p in paths]
    else:
        fs_loaders = fs_loaders_m
    fs_loaders_m = fs_loaders

    cold_lr = 1e-4
    alpha = 0.1 if 'alpha' not in params.keys() else params['alpha']

    task_num = len(fs_loaders[0].data) // task_size
    id_vals = list(set(fs_loaders[0].data[id_col]))
    n = len(id_vals)
    val_to_ind = dict(zip(id_vals, range(n)))

    p = np.ones(n)
    z = np.zeros(n)
    n_p_iter = 1000
    rho = params['rho']
    p_lr = params['p_lr']
    global_step = 0

    for epoch_i_ in range(n_epoch):

        # scheduler.step(epoch_i_)

        epoch_i = epoch_i_ % 3

        if epoch_i == 0:
            loader_a = fs_loaders[0]
            loader_b = fs_loaders[1]
        elif epoch_i == 1:
            loader_a = fs_loaders[2]
            loader_b = fs_loaders[0]
        else:
            loader_a = fs_loaders[1]
            loader_b = fs_loaders[2]

        p = calc_p(p, n_p_iter, p_lr, z, params['p_lr_decay'], rho)

        z = np.zeros(n)
        total_loss = 0.0

        # mini-bath
        batch_loss = 0.0
        batch_sum_p = 0.0
        batch_index = 0

        ###
        best_loss = 10
        best_auc = 0
        best_iter = 0
        count = 0
        ###

        # batch of tasks
        for i in tqdm(range(task_num), desc='epoch {} '.format(epoch_i_), mininterval=3):
            model.train()
            data_a, label_a = loader_a.load(i, cols, target_col=target_col)
            data_b, label_b = loader_b.load(i, cols, target_col=target_col)

            # a-setp
            y_a, meta_id_emb = model(data_a, warm=False)
            y_pred_a = y_a.flatten()
            y_true_a = torch.from_numpy(label_a).to(conf.device, dtype=torch.float)
            loss_a = criterion(y_pred_a, y_true_a)

            emb_grad = grad(loss_a, meta_id_emb, retain_graph=True)[0]
            meta_id_emb = meta_id_emb - cold_lr * emb_grad

            # b-step
            y_b, _ = model(data_a, warm=True, meta_emb=meta_id_emb)
            y_pred_b = y_b.flatten()
            y_true_b = torch.from_numpy(label_b).to(conf.device, dtype=torch.float)
            loss_b = criterion(y_pred_b, y_true_b)

            loss = (loss_a * alpha + loss_b * (1 - alpha))
            id_val = data_a[id_col].values[0]
            ind = val_to_ind[id_val]

            # calculate loss
            loss_with_p = p[ind] * loss
            z[ind] = loss

            # mini-bath
            batch_loss += loss_with_p
            batch_sum_p += p[ind]
            batch_index += 1

            if batch_index % batch_n_ID == 0:
                # update each batch
                batch_loss /= batch_sum_p

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                count = count + 1
                batch_loss = 0.0
                batch_sum_p = 0.0
                batch_index = 0

                # test_loss, test_auc = 0, 0
                test_loss, test_auc = test(model, warm=False)
                if params['gamma'] < 1.0:
                    scheduler.step(test_loss)

                ###
                if test_auc > best_auc:
                    best_auc = test_auc
                    best_loss = test_loss
                    best_iter = count

                ###
            total_loss += loss_with_p.item()

        if batch_loss > 0.0 and params['rest']:
            # update rest
            batch_loss /= batch_sum_p
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    print('best_iter', best_iter)
    print('best_auc', best_auc)
    print('best_loss', best_loss)
    print('end meta train')
    return model

@jit
def calc_p(p, n_p_iter, p_lr, z, p_lr_decay, rho):
    for j in range(n_p_iter):
        if j > 0 and j % 10 == 0:
            p_lr *= p_lr_decay
        p = p + p_lr * z
        p = project_onto_chi_square_ball(p, rho)
    return p