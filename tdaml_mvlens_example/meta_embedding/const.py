import logging

import torch
import torch.nn as nn

import meta_embedding.conf as conf
from meta_embedding.dataset import Loader
from meta_embedding.meta_model import MetaModel

from multiprocessing.dummy import Pool

pool = Pool(4)

id_col = 'MovieID'
item_col = ['Year']
context_col = ['Age', 'Gender', 'Occupation', 'UserID']
group_col = ['Title', 'Genres']
cols = [id_col] + item_col + context_col + group_col

num_words_dict = {
    'MovieID': 4000,
    'UserID': 6050,
    'Age': 7,
    'Gender': 2,
    'Occupation': 21,
    'Year': 83,
    'Title': 20001,
    'Genres': 21
}

padding_conf = {
    'Title': 8,
    'Genres': 4
}

# num_words_dict = {'aid': 2217,
#                   # 'uid': 82542667,
#                   'label': 2,
#                   'advertiserId': 158680,
#                   'campaignId': 766461,
#                   'creativeId': 1806761,
#                   'creativeSize': 110,
#                   'adCategoryId': 283,
#                   'productId': 28987,
#                   'productType': 12,
#                   'age': 6,
#                   'gender': 3,
#                   'marriageStatus': 16,
#                   'education': 8,
#                   'consumptionAbility': 3,
#                   'LBS': 998}
#
# padding_conf = {
# }

pre_train_path = 'data/big_train_main.pkl'
test_path = 'data/test_test.pkl'
# pre_train_path = 'data/data2/big_train.pkl'
# test_path = 'data/data2/test_partial_s.pkl'

criterion = nn.BCELoss()
criterion_sum = nn.BCELoss(reduction='sum')

t_loader = Loader(test_path, padding_conf)
target_col = 'y'
# target_col = 'label'
t_data, t_label = t_loader.load_all(cols, target_col=target_col)

# model_type = None
train_n_epoch = 3
test_n_epoch = 3

def new_model(path):
    model = MetaModel(id_col, item_col, context_col, group_col, conf.model_type, num_words_dict)
    model_dict = torch.load(path)
    model.load_state_dict(model_dict)
    return model.to(conf.device)

def new_logger():
    logger = logging.getLogger('/home/neo/PycharmProjects/aaai_experiment/log/log.txt')

    logger.setLevel(logging.INFO)

    fh = logging.FileHandler('spam.log')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_init_dist():
    import pickle
    import numpy as np
    from collections import Counter

    with open('data/big_train_main.pkl', "rb") as f:
        train = pickle.load(f)
    item_counter = Counter(train.MovieID)
    sort_items = item_counter.most_common()

    items, items_num = list(zip(*sort_items))
    items_num = np.array(items_num)
    items_prob = items_num / items_num.sum()

    return items_prob

logger = new_logger()