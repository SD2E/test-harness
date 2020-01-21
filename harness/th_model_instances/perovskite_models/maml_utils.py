import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
import pandas as pd
from copy import deepcopy
import pickle


def preprocess_chem_data(df_complete, cols_to_use, col_to_predict, k_shot, meta_batch_size=32, num_batches=100):
    # chem data is assumed binary output

    #df = pd.read_csv(path, usecols=cols_to_use)
    df = df_complete[cols_to_use]
    #df = df.rename(columns={split: renamed_split})
    print(df_complete[col_to_predict].unique())
    # df_complete[col_to_predict] = [1 if val ==
    #                               4 else 0 for val in df_complete[col_to_predict].values]
    # print(df_complete[col_to_predict].unique())
    amines = df_complete['_rxn_organic-inchikey'].unique().tolist()

    # Hold out 5 amines for testing, not going to use 1805 num because I feel like that's a lot of good
    # data for #learning
    #dont_holdout = "XFYICZOIWSBQSK-UHFFFAOYSA-N"
    #num_to_holdout = 1
    #hold_out_choices = [a for a in amines if a != dont_holdout]
    #hold_out_amines = np.random.choice(hold_out_choices, size=num_to_holdout)

    #amines = [a for a in amines if a not in hold_out_amines]

    batches = []
    for b in range(num_batches):
        x_spt, y_spt, x_qry, y_qry = [], [], [], []

        for mb in range(meta_batch_size):
            # grab task
            X = df_complete.loc[df_complete['_rxn_organic-inchikey']
                                == np.random.choice(amines)]
            X = X[cols_to_use]

            y = df_complete[col_to_predict].values
            X = X.values

            #scaler = StandardScaler()
            # scaler.fit(X)
            #X = scaler.transform(X)

            spt = np.random.choice(X.shape[0], size=k_shot, replace=False)
            qry = np.random.choice(X.shape[0], size=k_shot, replace=False)

            x_spt.append(X[spt])
            y_spt.append(y[spt])
            x_qry.append(X[qry])
            y_qry.append(y[qry])

        batches.append([np.array(x_spt), np.array(y_spt),
                        np.array(x_qry), np.array(y_qry)])

    return batches


def preprocess_pred_data(X, k_shot, cols_to_use, col_to_predict):
    assessment_batches = []
    x_spt, y_spt, x_qry, y_qry = [], [], [], []

    y = X[col_to_predict].values
    X = X[cols_to_use].values
    #X = X.drop(['_out_crystalscore', 'amine'], axis=1).values
    spt = np.random.choice(X.shape[0], size=k_shot, replace=False)
    qry = [i for i in range(len(X)) if i not in spt]
    if len(qry) <= 5:
        print("Warning: minimal testing data for meta-learn assessment")

    x_s = X[spt]
    y_s = y[spt]
    x_q = X[qry]
    y_q = y[qry]

    x_spt.append(x_s)
    y_spt.append(y_s)
    x_qry.append(x_q)
    y_qry.append(y_q)

    assessment_batches = [np.array(x_spt), np.array(
        y_spt), np.array(x_qry), np.array(y_qry)]
    # batches should have shape [num_batches, 4, meta_batch_size, k_shot, 68/1 - (68 features, 1 class values)]
    # not perfectly numpy array like, we're using some lists here
    return assessment_batches


class MAML:
    def __init__(self):
        super().__init__()

    def fit(self, train_df, cols):

        self.cols_to_use, self.col_to_predict = cols
        training_batches = preprocess_chem_data(train_df, self.cols_to_use,
                                                self.col_to_predict, k_shot=20,
                                                num_batches=250)

        input_dim = len(self.cols_to_use)
        config = [('linear', [400, input_dim]),
                  ('relu', [True]),
                  ('linear', [300, 400]),
                  ('relu', [True]),
                  ('linear', [200, 300]),
                  ('relu', [True]),
                  ('linear', [2, 200])]

        device = torch.device("cpu")
        self.maml = Meta(config, device, update_step_test=20, update_lr=0.2)

        for step in range(10001):
            b_num = np.random.choice(len(training_batches))
            batch = training_batches[b_num]

            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(batch[0]).float(), torch.from_numpy(
                batch[1]).long(), torch.from_numpy(batch[2]).float(), torch.from_numpy(batch[3]).long()

            _ = self.maml(x_spt, y_spt, x_qry, y_qry)

    def predict(self, X):
        testing_batches = preprocess_pred_data(
            X, k_shot=20, cols_to_use=self.cols_to_use, col_to_predict=self.col_to_predict)
        x_spt, y_spt, x_qry, y_qry = testing_batches[0], testing_batches[
            1], testing_batches[2], testing_batches[3]
        i = 0
        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
            preds = self.maml.finetunning(torch.from_numpy(x_spt_one).float(), torch.from_numpy(
                y_spt_one).long(), torch.from_numpy(x_qry_one).float(), torch.from_numpy(y_qry_one).long())
            pickle.dump(preds, open('result_{}'.format(i), 'w'))
            i += 1
        return preds


class Learner(nn.Module):
    def __init__(self, config):
        super(Learner, self).__init__()

        self.config = config
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(
                    torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(
                    torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (
                    param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (
                    param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv_transpose2d(
                    x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var,
                                 weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, config, device=None,
                 update_lr=0.3, meta_lr=1e-3,
                 n_way=2, k_spt=1, k_qry=15,
                 task_num=32, update_step=5,
                 update_step_test=3):
        super(Meta, self).__init__()

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.n_way = n_way
        self.k_spt = k_spt
        self.k_qry = k_qry
        self.task_num = task_num
        self.update_step = update_step
        self.update_step_test = update_step_test

        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_ = x_spt.size()
        querysz = x_qry.size(1)

        # losses_q[i] is the loss on step i
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            #print(x_spt.size(), y_spt.size())
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            # print(logits)
            # print(y_spt[i].shape)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                logits_q = self.net(
                    x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])

                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])

                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])

                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    # convert to numpy
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        #   print(torch.norm(p).item())
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        mccs = []
        prcs = []
        recalls = []
        aucs = []
        balanced_acc = []
        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                probs = F.softmax(logits_q, dim=1)
                pred_q = probs.argmax(dim=1)
                # convert to numpy
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[k + 1] = corrects[k + 1] + correct

                preds = pred_q.data.numpy()
                true_y = y_qry.data.numpy()
                #mccs.append(matthews_corrcoef(true_y, preds))
                #prcs.append(precision_score(true_y, preds))
                #recalls.append(recall_score(true_y, preds))
                #aucs.append(roc_auc_score(true_y, probs.data.numpy()[:, 1]))
                #balanced_acc.append(balanced_accuracy_score(true_y, preds))

        del net
        #accs = np.array(corrects) / querysz

        # return accs, mccs[-1], prcs[-1], recalls[-1], balanced_acc[-1]
        return preds
