import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Sigmoid

import meta_embedding.conf as conf


class MetaModel(nn.Module):

    class Flatten(nn.Module):
        def forward(self, input: torch.Tensor):
            return input.view(input.size(0), -1)

    def __init__(self, id_col, item_col,
                 context_col, group_col,
                 model_class, enum_counts,
                 emb_size=128):
        super(MetaModel, self).__init__()
        self.enum_counts = enum_counts
        self.emb_size = emb_size
        self.item_col = item_col
        self.group_col = group_col
        self.get_embeddings(id_col, item_col, context_col, group_col)
        self.get_meta_generator(len(item_col) + len(group_col))
        # self.get_meta_generator(len(item_col))

        n_cols = 1 + len(item_col) + \
                 len(context_col) + len(group_col)

        if model_class == 'DeepFM':
            self.get_deepFM(emb_size, n_cols)
            self.forward_ = self.forward_deepFM
        elif model_class == 'FM':
            self.get_FM(emb_size, n_cols, item_col, context_col, group_col)
            self.forward_ = self.forward_FM
        elif model_class == 'WD':
            self.get_WD(emb_size, n_cols, item_col, context_col, group_col)
            self.forward_ = self.forward_WD
        elif model_class == 'DNN':
            self.get_DNN(emb_size, n_cols)
            self.forward_ = self.forward_DNN

    def get_embeddings(self, id_col, item_cols, context_cols, group_cols):
        self.embs = nn.ModuleList([])
        columns = [id_col] + item_cols + context_cols + group_cols
        for col in columns:
            emb = nn.Embedding(self.enum_counts[col], self.emb_size, padding_idx=0)
            nn.init.xavier_uniform_(emb.weight.data)
            with torch.no_grad():
                emb.weight[0].fill_(0)
            self.embs.append(emb)

    def get_meta_generator(self, n_item_col):
        self.generator = nn.Sequential(
            MetaModel.Flatten(),
            nn.Linear(self.emb_size * n_item_col,
                      self.emb_size, bias=False),
            nn.Tanh()
        )

    def get_FM(self, emb_size, num_col, item_col, context_col, group_col):
        self.ws = nn.ModuleList([])
        columns = item_col + context_col + group_col
        for col in columns:
            w = nn.Embedding(self.enum_counts[col], 1, padding_idx=0)
            nn.init.xavier_uniform_(w.weight.data)
            with torch.no_grad():
                w.weight[0].fill_(0)
            self.ws.append(w)

        self.out = nn.Sigmoid()
        self.merge_output = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def get_WD(self, emb_size, num_col, item_col, context_col, group_col):

        self.ws = nn.ModuleList([])
        columns = item_col + context_col + group_col
        for col in columns:
            w = nn.Embedding(self.enum_counts[col], 1, padding_idx=0)
            nn.init.xavier_uniform_(w.weight.data)
            with torch.no_grad():
                w.weight[0].fill_(0)
            self.ws.append(w)

        element_num = emb_size * num_col
        self.deep_part = nn.Sequential(
            nn.Linear(element_num, element_num),
            nn.ReLU(),
            nn.Linear(element_num, element_num),
            nn.ReLU(),
            nn.Linear(element_num, 1),
        )
        self.merge_output = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def get_deepFM(self, emb_size, num_col):
        element_num = emb_size * num_col
        self.dnn = nn.Sequential(
            nn.Linear(element_num, element_num),
            nn.ReLU(),
            nn.Linear(element_num, element_num),
            nn.ReLU()
        )
        self.merge_output = nn.Sequential(
            nn.Linear(num_col + element_num, 1),
            nn.Sigmoid()
        )

    def get_DNN(self, emb_size, num_col):
        element_num = emb_size * num_col
        self.dnn = nn.Sequential(
            nn.Linear(element_num, element_num),
            nn.ReLU(),
            nn.Linear(element_num, element_num),
            nn.ReLU(),
            nn.Linear(element_num, 1),
            nn.Sigmoid()
        )

    def forward_meta_generator(self, item_embs: list):
        frozen = torch.stack(item_embs, dim=1).detach()
        pred_emb = self.generator(frozen) / 5
        return pred_emb

    def forward_FM(self, data: pd.DataFrame, meta_id_emb):

        emb_vals = []
        wide_vals = []

        for i, col in enumerate(data):
            # special case: training meta.
            if i == 0 and meta_id_emb is not None:
                emb_vals.append(meta_id_emb)
            else:
                data_col = torch.cuda.LongTensor(data[col].to_list(), device=conf.device)

                if i > 0:
                    wide_val = self.ws[i - 1](data_col)
                    if len(wide_val.shape) > 2:
                        wide_val = wide_val.squeeze(-1)
                    wide_vals.append(wide_val)

                if col in self.group_col:
                    emb_val = self.embs[i](data_col)
                    emb_vals.append(emb_val.mean(dim=1))
                else:
                    emb_vals.append(self.embs[i](data_col))

        h1 = torch.cat(wide_vals, dim=1).sum(dim=1, keepdim=True)
        h2 = torch.cat(emb_vals, dim=1).sum(dim=1, keepdim=True)
        output = self.merge_output(torch.cat([h1, h2], dim=1))
        return output

    def forward_WD(self, data: pd.DataFrame, meta_id_emb):

        emb_vals = []
        wide_vals = []

        for i, col in enumerate(data):
            # special case: training meta.
            if i == 0 and meta_id_emb is not None:
                emb_vals.append(meta_id_emb)
            else:
                data_col = torch.cuda.LongTensor(data[col].to_list(), device=conf.device)

                if i > 0:
                    wide_val = self.ws[i - 1](data_col)
                    if len(wide_val.shape) > 2:
                        wide_val = wide_val.squeeze(-1)
                    wide_vals.append(wide_val)

                if col in self.group_col:
                    emb_val = self.embs[i](data_col)
                    emb_vals.append(emb_val.mean(dim=1))
                else:
                    emb_vals.append(self.embs[i](data_col))

        sum_embs = sum(emb_vals)
        diff_emb_vals = [sum_embs - val for val in emb_vals]
        dot_emb_vals = [torch.sum(v * d, dim=1, keepdim=True) for v, d in zip(emb_vals, diff_emb_vals)]

        h1 = torch.cat(wide_vals, dim=1).sum(dim=1, keepdim=True)
        h2 = torch.cat(dot_emb_vals, dim=1).sum(dim=1, keepdim=True)
        output = self.merge_output(torch.cat([h1, h2], dim=1))

        return output

    def forward_deepFM(self, data: pd.DataFrame, meta_id_emb):

        emb_vals = []

        for i, col in enumerate(data):
            # special case: training meta.
            if i == 0 and meta_id_emb is not None:
                emb_vals.append(meta_id_emb)
            else:
                try:
                    data_col = torch.cuda.LongTensor(data[col].to_list(), device=conf.device)
                except:
                    print('?')
                if col in self.group_col:
                    emb_val = self.embs[i](data_col)
                    emb_vals.append(emb_val.mean(dim=1))
                else:
                    emb_vals.append(self.embs[i](data_col))

        sum_embs = sum(emb_vals)
        diff_emb_vals = [sum_embs - val for val in emb_vals]
        dot_emb_vals = [torch.sum(v * d, dim=1, keepdim=True) for v, d in zip(emb_vals, diff_emb_vals)]

        h1 = torch.cat(dot_emb_vals, dim=1)
        h2 = self.dnn(torch.cat(emb_vals, dim=1))
        output = self.merge_output(torch.cat([h1, h2], dim=1))

        return output

    def forward_DNN(self, data: pd.DataFrame, meta_id_emb):
        emb_vals = []
        for i, col in enumerate(data):
            # special case: training meta.
            if i == 0 and meta_id_emb is not None:
                emb_vals.append(meta_id_emb)
            else:
                data_col = torch.cuda.LongTensor(data[col].to_list(), device=conf.device)
                if col in self.group_col:
                    emb_val = self.embs[i](data_col)
                    emb_vals.append(emb_val.mean(dim=1))
                else:
                    emb_vals.append(self.embs[i](data_col))

        output = self.dnn(torch.cat(emb_vals, dim=1))

        return output

    def forward(self, data, warm=True, meta_emb=None, emb_only=False):
        if not warm:
            item_embs = []

            for i, col in enumerate(data):
                if col in self.group_col:
                    data_col = torch.cuda.LongTensor(data[col].to_list(), device=conf.device)
                    emb_val = self.embs[i](data_col)
                    item_embs.append(emb_val.mean(dim=1))
                elif col in self.item_col:
                    data_col = torch.cuda.LongTensor(data[col].to_list(), device=conf.device)
                    item_embs.append(self.embs[i](data_col))

            meta_id_emb = self.forward_meta_generator(item_embs)
            if emb_only:
                return None, meta_id_emb
            else:
                return self.forward_(data, meta_id_emb), meta_id_emb

        if meta_emb is not None:
            return self.forward_(data, meta_emb), None

        return self.forward_(data, meta_id_emb=None), None

    def set_id_emb(self, id_val, id_emb):
        self.embs[0].weight.data[id_val] = id_emb.detach()
