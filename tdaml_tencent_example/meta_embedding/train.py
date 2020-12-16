import torch.nn
from torch.autograd import grad
# from tensorboardX import SummaryWriter
from meta_embedding.test import *
# import torch.optim.lr_scheduler as lr_scheduler

def pre_train():

    print('[pre train]')
    model = MetaModel(id_col, item_col, context_col, group_col, conf.model_type, num_words_dict)
    model.to(conf.device)
    ps = conf.model_type_params.get(conf.model_type)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=ps['base_lr'], weight_decay=ps['base_penalty'])
    batch_size = 200
    data_loader = BatchLoader(pre_train_path, padding_conf, batch_size)
    epoch = 1
    batch_num = len(data_loader.data) // batch_size

    # model.train()
    # total_loss = 0

    for epoch_i in range(epoch):

        model.train()

        for i in tqdm(range(batch_num), desc='epoch:{}'.format(epoch_i)):
            # make a prediction
            batch_x, batch_y = data_loader.load(i, cols, target_col=target_col)
            y, _ = model(batch_x)
            y_pred = y.flatten()
            y_true = torch.from_numpy(batch_y).float().to(conf.device)
            loss = criterion(y_pred, y_true)

            # back prop.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def meta_train(model, n_epoch=3):
    print('[meta-train]')

    model = new_model(model) if isinstance(model, str) else model
    paths = ['data/train_oneshot_a.pkl',
             'data/train_oneshot_b.pkl',
             'data/train_oneshot_c.pkl']
    ps = conf.model_type_params.get(conf.model_type)

    optimizer = torch.optim.Adam(params=model.generator.parameters(), lr=ps['train_lr'], weight_decay=ps['penalty'])
    task_size = 200
    batch_n_ID = 50
    batch_size = task_size * batch_n_ID

    fs_loaders = [BatchLoader(p, padding_conf, batch_size) for p in paths]
    # fs_loaders = pool.map(lambda p: BatchLoader(p, padding_conf, batch_size), paths, 1)
    cold_lr = 1e-4
    alpha = ps['alpha']

    for epoch_i_ in range(n_epoch):
        model.train()
        epoch_i = epoch_i_ % 3
        if epoch_i == 0:
            loader_a = fs_loaders[1]
            loader_b = fs_loaders[0]
        elif epoch_i == 1:
            loader_a = fs_loaders[2]
            loader_b = fs_loaders[1]
        else:
            loader_a = fs_loaders[0]
            loader_b = fs_loaders[2]

        batch_num = len(loader_a.data) // batch_size
        # batch of tasks
        for i in tqdm(range(batch_num), desc='epoch {} '.format(epoch_i_)):
            data_a, label_a = loader_a.load(i, cols, target_col=target_col)
            data_b, label_b = loader_b.load(i, cols, target_col=target_col)

            # a-setp
            y_a, meta_id_emb = model(data_a, warm=False)

            loss_a = criterion(
                y_a.flatten(),
                torch.from_numpy(label_a).to(conf.device, dtype=torch.float))

            # inner train
            emb_grad = grad(loss_a, meta_id_emb, retain_graph=True)[0]
            meta_id_emb = meta_id_emb - cold_lr * emb_grad

            # b-step
            y_b, _ = model(data_a, warm=True, meta_emb=meta_id_emb)

            loss_b = criterion(
                y_b.flatten(),
                torch.from_numpy(label_b).to(conf.device, dtype=torch.float))

            # back prop.
            loss = loss_a * alpha + loss_b * (1 - alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        l, a = test(model, warm=False)
        print('loss:{}, auc:{}'.format(round(l, 5), round(a, 5)))

    print('end meta train')
    return model
