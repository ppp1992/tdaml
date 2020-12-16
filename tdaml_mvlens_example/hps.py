from hyperopt import hp

models = {

    'DNN': {
        'lr': hp.choice('lr', [0.01]),
        'weight_decay': hp.choice('weight_decay', [0]),
        'amsgrad': hp.choice('amsgrad', [False]),
        'batch_n_ID': hp.choice('batch_n_ID', [400]),
        'rho': hp.choice('rho', [0.00001]),
        'p_lr': hp.choice('p_lr', [0.0001]),
        'p_lr_decay': hp.choice('p_lr_decay', [1.0]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.1]),
    },

    'DeepFM': {
        'lr': hp.choice('lr', [0.01]),
        'weight_decay': hp.choice('weight_decay', [0]),
        'amsgrad': hp.choice('amsgrad', [False]),
        'batch_n_ID': hp.choice('batch_n_ID', [200]),
        'rho': hp.choice('rho', [0.0001]),
        'p_lr': hp.choice('p_lr', [0.0001]),
        'p_lr_decay': hp.choice('p_lr_decay', [0.05]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.7]),
    },
    'WD': {
        'lr': hp.choice('lr', [0.01]),
        'weight_decay': hp.choice('weight_decay', [1e-5]),
        'amsgrad': hp.choice('amsgrad', [False]),
        'batch_n_ID': hp.choice('batch_n_ID', [25]),
        'rho': hp.choice('rho', [0.001]),
        'p_lr': hp.choice('p_lr', [0.001]),
        'p_lr_decay': hp.choice('p_lr_decay', [1.0]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.5]),
    },

    'FM': {
        'lr': hp.choice('lr', [0.01]),
        'weight_decay': hp.choice('weight_decay', [1e-05]),
        'amsgrad': hp.choice('amsgrad', [1e-05]),
        'batch_n_ID': hp.choice('batch_n_ID', [25]),
        'rho': hp.choice('rho', [0.001]),
        'p_lr': hp.choice('p_lr', [0.001]),
        'p_lr_decay': hp.choice('p_lr_decay', [0.05]),
        'gamma': hp.choice('gamma', [1.0]),
        'alpha': hp.choice('alpha', [0.5]),
    },
}
