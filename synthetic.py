from torch.optim import lr_scheduler
from torch import nn
import torch
import numpy as np
import random
import os
import xlwt
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patience = 5
batch_size = 2048
max_iterations = 32768*128//batch_size
stop_check = 128*128//batch_size
valid_iter = 1024*128//batch_size
LR_step_size = 8192*8//batch_size
gamma = 0.7
LR = 0.1


order_of_poly = 4

f = xlwt.Workbook()
sheet = f.add_sheet('AnyCubic', cell_overwrite_ok=True)
dims = [8]
h_dim = [8, 16, 32, 64]
n_layers = [2, 3, 4, 5]
in_range = 1
wei_range = 5
m_names = ["ReLuln", "Sigln", "SPln", "Pac2ln", "Pac3ln",
           "ReLurn", "Sigrn", "SPrn", "Pac2rn", "Pac3rn",
          ]

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.state_dict = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.update_state(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_state(val_loss, model)
            self.counter = 0

    def update_state(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.state_dict = model.state_dict()
        self.val_loss_min = val_loss


def ChSh3(x):
    return 4*x**3-3*x


def ChSh2(x):
    return 2*x**2-1


class RangeNorm(nn.Module):
    def __init__(self, dim):
        super(RangeNorm, self).__init__()
        self.dim = dim
    def forward(self, x):
        min_ = x.min(dim=-1, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0]
        return (2*x-(min_+max_))/(max_-min_)


class FNN(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layer, act, norm=None):
        super(FNN, self).__init__()
        self.dims = hid_dim
        self.n_layer = num_layer
        self.fnn = [nn.Linear(input_dim, self.dims)]
        for _ in range(num_layer-1):
            self.fnn.append(nn.Linear(self.dims, self.dims))
        self.fnn = nn.ModuleList(self.fnn)
        if norm == 'LN':
            self.norm = [nn.LayerNorm([self.dims]) for _ in range(num_layer)]
        elif norm == 'BN':
            self.norm = [nn.BatchNorm1d(self.dims) for _ in range(num_layer)]
        elif norm == 'RN':
            self.norm = [RangeNorm(self.dims) for _ in range(num_layer)]
        else:
            self.norm = [lambda x: x for _ in range(num_layer)]
        self.norm = nn.ModuleList(self.norm)
        self.act = act  # [act for _ in range(num_layer)]
        self.fc = nn.Linear(self.dims, 1)
        self.input_dim = input_dim

    def forward(self, x):
        x = self.act(self.norm[0](self.fnn[0](x)))
        for i in range(1, self.n_layer):
            x = self.act(self.norm[i](self.fnn[i](x)))+x
        return self.fc(x)


results = np.zeros((len(m_names), len(h_dim), len(n_layers), 5))

for i in range(len(m_names)):
    # start_n = i*len(dims)*len(in_ranges)*len(wei_ranges)
    start_n = 0
    sheet.write(i*10, start_n, m_names[i])
    for j in range(len(dims)):
        start_d = start_n+j*len(h_dim)*len(n_layers)
        sheet.write(i*10+1, start_d, str(dims[j]))
        for k in range(len(h_dim)):
            start_ir = start_d+k*len(n_layers)
            sheet.write(i*10+2, start_ir, str(h_dim[k]))
            for l in range(len(n_layers)):
                sheet.write(i*10+3, start_ir+l, str(n_layers[l]))

for d_idx in range(len(dims)):
    d_ = dims[d_idx]
    print("input dims:", d_)

    for nl_idx in range(len(n_layers)):
        n_layer = n_layers[nl_idx]
        print("number of layers:", n_layer)

        for h_idx in range(len(h_dim)):
            h_d = h_dim[h_idx]
            print("hidden dims:", h_d)

            models, init_stats, optimizers = [], [], []
            for m_dix in range(len(m_names)):
                m_name = m_names[m_dix]
                if 'rn' in m_name:
                    ln = 'RN'
                elif 'ln' in m_name:
                    ln = 'LN'
                elif 'bn' in m_name:
                    ln = 'BN'
                else:
                    ln = None
                if 'Pac3' in m_name:
                    main_model = FNN(d_, h_d, n_layer, ChSh3, norm=ln).to(device)
                elif 'Pac2' in m_name:
                    main_model = FNN(d_, h_d, n_layer, ChSh2, norm=ln).to(device)
                elif 'SP' in m_name:
                    main_model = FNN(d_, h_d, n_layer, nn.Softplus(), norm=ln).to(device)
                elif 'Sig' in m_name:
                    main_model = FNN(d_, h_d, n_layer, nn.Sigmoid(), norm=ln).to(device)
                elif 'ReLu' in m_name:
                    main_model = FNN(d_, h_d, n_layer, nn.ReLU(), norm=ln).to(device)
                models.append(main_model)
                optimizers.append(torch.optim.Adam(models[m_dix].parameters(), lr=LR))  # Adadelta
                init_stats.append(models[m_dix].state_dict())

            for idx in range(5):
                    print("round", idx)
                    fuc = [np.random.uniform(-wei_range, wei_range, 1)]
                    fuc.extend(
                        [np.random.uniform(-wei_range, wei_range, dims) for dims in
                        [[d_ for _ in range(order)] for order in range(1, order_of_poly+1)]]
                    )

                    def add_k(idxs, c_order, t_order, weights, in_data):
                        # compute the c_order to t_order terms given c_order-1 fixed indices
                        value = 0
                        if c_order <= t_order:
                            for i in range(idxs, d_):
                                value += weights[0][i] * in_data[:, i]
                                # fix the next index
                                value = value + add_k(i, c_order+1, t_order,
                                                [weights[w_l+1][i] for w_l in range(len(weights)-1)],
                                                in_data[:]*in_data[:, i:i+1])
                        return value

                    def gen_data():
                        # generate the polynomial output given d_ inputs
                        data_ = np.random.uniform(-1, 1, (batch_size, d_))
                        label = fuc[0]*np.ones(batch_size)
                        label = label + add_k(0, 1, order_of_poly, fuc[1:], data_)
                        return data_[:, :], label[:, np.newaxis]

                    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)
                    early_stoppings = [EarlyStopping(patience) for _ in range(len(m_names))]
                    train_losses = [0 for _ in range(len(m_names))]
                    for m_idx in range(len(m_names)):
                        models[m_idx].load_state_dict(init_stats[m_idx])

                    for itera in range(1, max_iterations+1):
                        train_data, train_label = gen_data()
                        train_data = torch.from_numpy(train_data).float().to(device)
                        train_label = torch.from_numpy(train_label).float().to(device)
                        stop = True
                        for m_idx in range(len(m_names)):
                            if early_stoppings[m_idx].early_stop:
                                continue
                            else:
                                stop = False
                            models[m_idx].train()
                            optimizers[m_idx].zero_grad()

                            output = models[m_idx](train_data)
                            loss_fn = torch.nn.MSELoss()
                            loss = loss_fn(output, train_label)
                            loss.backward()

                            optimizers[m_idx].step()
                            # exp_lr_scheduler.step()
                            train_losses[m_idx] += loss.item()
                            if itera % stop_check == 0:
                                # print(itera, "loss:", train_loss/stop_check)
                                early_stoppings[m_idx](train_losses[m_idx]/stop_check, models[m_idx])
                                if early_stoppings[m_idx].early_stop:
                                    # print("Early stopping at iter", itera)
                                    continue
                                train_losses[m_idx] = 0
                        if stop:
                            break

                    for m_idx in range(len(m_names)):
                        models[m_idx].load_state_dict(early_stoppings[m_idx].state_dict)
                    test_losses = [0 for _ in range(len(m_names))]

                    with torch.no_grad():
                        for m_idx in range(len(m_names)):
                            models[m_idx].eval()
                        for itera in range(valid_iter):
                            test_data, test_label = gen_data()
                            test_data = torch.from_numpy(test_data).float().to(device)
                            test_label = torch.from_numpy(test_label).float().to(device)

                            for m_idx in range(len(m_names)):
                                output = models[m_idx](test_data)
                                loss_fn = torch.nn.MSELoss()
                                loss = loss_fn(output, test_label)

                                test_losses[m_idx] += loss.item()
                        for m_idx in range(len(m_names)):
                            print("model:" + m_names[m_idx])
                            print("test loss:", test_losses[m_idx]/valid_iter)
                            sheet.write(4+idx+m_idx*10,
                                        nl_idx + len(n_layers) * (h_idx + len(h_dim)*d_idx),
                                        test_losses[m_idx]/valid_iter)
                            results[m_idx, h_idx, nl_idx, idx] = test_losses[m_idx]/valid_iter
results = results.mean(axis=-1)
results = np.reshape(results, (len(m_names), -1))
rank_idx = np.argsort(results, axis=0)
rank = np.argsort(rank_idx, axis=0)


sheet.write(2 + len(m_names) * 10, 0, 'avg loss')
sheet.write(14 + len(m_names) * 10, 0, 'ranks')
sheet.write(14 + len(m_names) * 10, len(h_dim)*len(n_layers),
                'avg rank')
for m_idx in range(len(m_names)):
    for row in range(len(h_dim)*len(n_layers)):
        sheet.write(3+len(m_names)*10+m_idx, row,
                    results[m_idx, row])

        sheet.write(15+len(m_names)*10+m_idx, row,
                    int(rank[m_idx, row])+1)
    sheet.write(15 + len(m_names) * 10 + m_idx, len(h_dim)*len(n_layers),
                rank[m_idx].mean()+1.0)

f.save('synthetic_polynomial_of_degree_'+str(order_of_poly)+'.xls')
for m_idx in range(len(m_names)):
    print(m_names[m_idx], end='&')
    for row in range(len(h_dim)*len(n_layers)):
        if rank_idx[0, row] == m_idx:
            print('\\textbf{%.3f}&' % results[m_idx, row], end='')
        elif rank_idx[1, row] == m_idx:
            print('\\underline{%.3f}&' % results[m_idx, row], end='')
        else:
            print('%.3f&' % results[m_idx, row], end='')
    print('%.2f' % (rank[m_idx].mean()+1.0))
