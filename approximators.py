# Import packages
import torch
import numpy as np
import time
from numpy.linalg import inv
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
# Import implementations in folder
from models import twin_nn
from diffPCA import DiffPCA


class BespokeDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.input = x_data
        self.truth = y_data

    def __getitem__(self, index):
        return self.input[index], self.truth[index]

    def __len__(self):
        return len(self.input)


def data_normalizer(x, y, xbar, eps=1e-8):
    # Normalize inputs
    x_mean = x.mean(axis=0)
    x_sd = x.std(axis=0) + eps
    x_out = (x - x_mean)/x_sd

    # Normalize payoff labels
    y_mean = y.mean(axis=0)
    y_sd = y.std(axis=0) + eps
    y_out = (y - y_mean)/y_sd

    # Normalize differential labels and prepare adjustment factors
    if xbar is not None:
        xbar_out = xbar*(x_sd/y_sd)
        xbar_norm = 1 / np.sqrt((xbar_out**2).mean(axis=0)).reshape(1, -1)
    else:
        xbar_out = None
        xbar_norm = None
    return x_mean, x_sd, x_out, y_mean, y_sd, y_out, xbar_out, xbar_norm


class nn_approximator:
    def __init__(self, x_raw, y_raw, xbar_raw):
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.xbar_raw = xbar_raw
        self.model = None

    # Initialize neural net and prepare data
    def prepare(self, n_hidden_layers=4,
                n_units=20,
                differential=True,
                PCA=False,
                device='cpu',
                seed=1,
                silent=True):
        torch.manual_seed(seed)
        self.differential = differential
        self.device = device
        self.PCA = PCA
        self.x_mean, self.x_sd, self.x, self.y_mean, self.y_sd, self.y, self.xbar, self.xbar_norm = \
            data_normalizer(self.x_raw, self.y_raw, self.xbar_raw)
        if self.PCA:
            tmp = time.time()
            old_dim = self.x.shape[1]
            self.PCA_obj = DiffPCA(1.0e-8, differential=self.differential)
            self.x, self.xbar = self.PCA_obj.fit_transform(self.x, self.xbar)
            self.xbar_norm = np.repeat(1, self.x.shape[1])
            self.PCA_time = round(time.time() - tmp, 3)
            if not silent:
                print('PCA step reduced dimensionality from {} to {}'.format(old_dim, self.x.shape[1]))
        self.nFeatures = self.x.shape[1]
        self.model = twin_nn(n_hidden_layers, n_units, self.nFeatures, self.differential)

    def train(self, epochs, batch_size, learning_rate, lam=1, schedule_lr=True, verbose=True, silent=False):
        if silent:
            verbose = False
        tmp = time.time()
        # Train model
        if not silent:
            print("--------- Training ---------")

        train_data = BespokeDataset(self.x, np.column_stack((self.y, self.xbar)))
        train_loader = DataLoader(train_data, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if schedule_lr:
            scheduler = OneCycleLR(optimizer,
                                   max_lr=learning_rate,
                                   steps_per_epoch=len(train_loader),
                                   epochs=epochs,
                                   pct_start=0.2)
        else:
            scheduler = None

        training_loss = []
        self.model.to(self.device)
        self.model.train()
        for e in range(0, epochs):
            epoch_loss = 0
            n_minibatches = 0
            for input_train_batch, truth_train_batch in train_loader:
                x = input_train_batch.to(self.device).float()
                truth_train_batch_tmp = truth_train_batch.to(self.device).float()
                z_true, xbar_true = truth_train_batch_tmp[:, 0].unsqueeze(1), truth_train_batch_tmp[:, 1:]
                optimizer.zero_grad()
                z_pred, xbar_pred = self.model(x)  # produces predictions on training batch
                mse_loss = nn.MSELoss()
                if self.model.differential:
                    loss = mse_loss(z_true, z_pred) + lam*self.nFeatures*mse_loss(xbar_true * torch.Tensor(self.xbar_norm), xbar_pred * torch.Tensor(self.xbar_norm))
                else:
                    loss = mse_loss(z_true, z_pred)

                loss.backward()  # initiate backprop through entire (twin) network
                optimizer.step()  # update weights and biases
                if scheduler is not None:
                    scheduler.step()

                epoch_loss += loss.item()
                n_minibatches += 1

            t_loss = epoch_loss / n_minibatches

            training_loss.append(t_loss)
            if verbose:
                print('EPOCH: %s | training loss: %s ' % (e + 1, round(t_loss, 5)))

        self.training_time = round(time.time() - tmp, 3)
        if not silent:
            print("---------- Done! ----------")
            print(self.device, "training time: %s seconds" % (str(self.training_time)))

    # Predict
    def predict(self, xs):
        scaled_xs = (xs - self.x_mean)/self.x_sd
        if self.PCA:
            scaled_xs = self.PCA_obj.transform(scaled_xs)
        pred_data = BespokeDataset(scaled_xs, scaled_xs) # note: second parameter will not be used here
        prediction_loader = DataLoader(pred_data, len(pred_data))
        self.model.eval()
        for input_pred_batch, _ in prediction_loader:
            x = input_pred_batch.to(self.device).reshape(len(pred_data), -1).float()
            y_pred, _ = self.model(x)
            xbar_pred = self.model.backprop(y_pred, x)
        y_pred = y_pred.detach().numpy()
        xbar_pred = xbar_pred.detach().numpy()
        if self.PCA:
            xbar_pred = self.PCA_obj.inverse_transform(xbar_pred)
        y_pred = y_pred*self.y_sd + self.y_mean
        xbar_pred = xbar_pred*self.y_sd/self.x_sd
        return y_pred, xbar_pred


class poly_approximator:
    def __init__(self, x_raw, y_raw, xbar_raw):
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.xbar_raw = xbar_raw
        if x_raw.shape[1] != 1:
            raise ValueError('polynomial regression only applicable for one-dimensional assets')

    def prepare(self, d, differential=True):
        # polynomial degree
        self.d = d
        self.x = self.x_raw
        self.y = self.y_raw
        self.xbar = self.xbar_raw
        self.differential = differential

    def train(self, lam=1, silent=False):
        # Fit model
        tmp = time.time()
        self.A = np.array([np.power(s, np.arange(self.d+1)) for s in self.x])
        self.B = np.zeros_like(self.A)
        for i, s in enumerate(self.x):
            for j in range(1, self.d+1):
                self.B[i, j] = j*np.power(s, j-1)
        if self.differential:
            alpha = 1/(1+lam)
            beta = 1-alpha
            theta1 = inv(alpha*np.matmul(self.A.T, self.A) + beta*np.matmul(self.B.T, self.B))
            theta2 = alpha*np.matmul(self.A.T, self.y) + beta*np.matmul(self.B.T, self.xbar)
            self.theta = np.matmul(theta1, theta2)
        else:
            gram = np.matmul(self.A.T, self.A)
            moment = np.matmul(self.A.T, self.y)
            self.theta = np.matmul(inv(gram), moment)

        self.training_time = round(time.time() - tmp, 3)
        if not silent:
            print("Fitted polynomial regression in: %s seconds" % (str(self.training_time)))

    def predict(self, xs):
        xs = xs.reshape((-1, 1))
        # predict prices
        price_func = lambda s: np.sum(self.theta.reshape((-1,))*np.power(s, np.arange(self.d + 1)))
        vec_price_func = np.vectorize(price_func)
        y_pred = vec_price_func(xs)
        # predict differentials
        diff_func = lambda s: np.sum(np.arange(1,self.d+1)*self.theta[1:].reshape((-1,))*np.power(s, np.arange(self.d)))
        vec_diff_func = np.vectorize(diff_func)
        xbar_pred = vec_diff_func(xs)
        return y_pred, xbar_pred
