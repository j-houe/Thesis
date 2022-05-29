import numpy as np
import torch
import time
from numpy import exp, log
from numpy.random import normal
from scipy.stats import norm
from torch import Tensor
from payoffs import Payoff


class Generator:
    def sim_paths(self, *args, **kwargs):
        raise NotImplementedError("method simPaths must be implemented")

    def test_set(self, *args, **kwargs):
        raise NotImplementedError("method test_set must be implemented")

    def update_sigmaAdj(self, new_sigmaAdj):
        self.sigmaAdj = new_sigmaAdj

    def update_nStocks(self, new_nStocks):
        self.nStocks = new_nStocks


class BlackScholes(Generator):
    def __init__(self, sigma=0.2, r=0, K=1.1, S0=1, T1=1, T2=2, sigmaAdj=1.5):
        self.sigma = sigma
        self.r = r
        self.K = K
        self.S0 = S0
        self.T1 = T1
        self.T2 = T2
        self.sigmaAdj = sigmaAdj

    def sim_paths(self, nPaths, seed=1, antithetic=True):
        tmp = time.time()
        np.random.seed(seed)

        # Simulate paths of underlying
        normals = normal(size=[nPaths, 2])
        drift1 = (self.r - 0.5*self.sigma**2)*self.T1
        drift2 = (self.r - 0.5*self.sigma**2)*(self.T2 - self.T1)
        vol1 = self.sigmaAdj*self.sigma*np.sqrt(self.T1)
        vol2 = self.sigma * np.sqrt(self.T2 - self.T1)
        S1 = self.S0*exp(drift1 + vol1*normals[:, 0])
        S2 = S1*exp(drift2 + vol2*normals[:, 1])

        # Evaluate time t=1 value of time t=2 call payoff
        payoff = exp(-self.r*(self.T2-self.T1))*np.maximum(S2-self.K, 0)

        # Compute pathwise differentials
        deltas = np.where(S2 > self.K, 1, 0)*S2/S1

        # Optional antithetic sampling. Default is true
        if antithetic:
            S2_anti = S1*exp(drift2 - vol2*normals[:, 1])
            payoff_anti = exp(-self.r*(self.T2-self.T1))*np.maximum(S2_anti-self.K, 0)
            deltas_anti = np.where(S2_anti > self.K, 1, 0)*S2_anti/S1

            # Prepare output
            x = S1
            y = 0.5*(payoff + payoff_anti)
            xbar = 0.5*(deltas + deltas_anti)
        else:
            # Prepare output
            x = S1
            y = payoff
            xbar = deltas
        self.sim_time = round(time.time() - tmp, 3)
        return x.reshape((-1, 1)), y.reshape((-1, 1)), xbar.reshape((-1, 1))

    def true_price(self, S1_input):
        vol = self.sigma * np.sqrt(self.T2 - self.T1)
        d1 = (log(S1_input / self.K) + (self.r + 0.5 * self.sigma ** 2) * (self.T2 - self.T1)) / vol
        d2 = d1 - vol
        return norm.cdf(d1) * S1_input - norm.cdf(d2) * self.K * exp(-self.r * (self.T2 - self.T1))

    def true_delta(self, S1_input):
        vol = self.sigma * np.sqrt(self.T2 - self.T1)
        d1 = (log(S1_input / self.K) + (self.r + 0.5 * self.sigma ** 2) * (self.T2 - self.T1)) / vol
        return exp(-self.r * (self.T2 - self.T1)) * norm.cdf(d1)

    def test_set(self, lower=0.5, upper=1.50, num=100, seed=None):
        spots = np.linspace(lower, upper, num)
        # Compute benchmark price+delta
        true_y = self.true_price(spots)
        true_delta = self.true_delta(spots)
        return spots, spots, true_y.reshape((-1, 1)), true_delta.reshape((-1, 1))


class SABR(Generator):
    # We introduce a floor at 0 when updating S in the Euler scheme since NAs would appear for negative S and beta<1.
    # More sophisticated methods exists to accommodate negative asset values but this suffices for our
    # proof-of-concept examples.
    def __init__(self, payoff: Payoff,
                 S0=1, sigma0=0.2, T1=1, T2=2, K=1.1, alpha=0.5, beta=0.5, rho=-0.5, sigmaAdj=1.5):
        self.payoff = payoff
        self.S0 = S0
        self.sigma0 = sigma0
        self.T1 = T1
        self.T2 = T2
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.sigmaAdj = sigmaAdj

    def sim_paths(self, nPaths, nSteps=100, seed=None, antithetic=True):
        tmp = time.time()
        if seed is None:
            torch.manual_seed(np.random.randint(1e6))
        else:
            torch.manual_seed(seed)
        asset_normals = torch.normal(0, 1, size=[2, nPaths, nSteps, 1])
        vol_normals = torch.normal(0, 1, size=[2, nPaths, nSteps, 1])
        stepSize1 = Tensor([self.T1/nSteps])
        stepSize2 = Tensor([(self.T2 - self.T1)/nSteps])

        # Simulate exposure date paths
        S1 = Tensor(np.repeat(self.S0, nPaths))
        sigma = self.sigma0*self.sigmaAdj
        for i in range(nSteps):
            W1 = asset_normals[0, :, i, :].reshape((-1))
            W2 = self.rho*W1 + np.sqrt(1-self.rho*self.rho)*vol_normals[0, :, i, :].reshape(-1)
            # Update S
            S1 = torch.maximum(S1 + sigma*torch.pow(S1, self.beta)*torch.sqrt(stepSize1)*W1, Tensor([0]))
            # Update sigma
            sigma = sigma*exp(-0.5*self.alpha*self.alpha*stepSize1 + self.alpha * np.sqrt(stepSize1)*W2)
        self.S1 = S1

        # Simulate payoff date paths
        S2 = self.S1
        S2.requires_grad = True
        sigma = self.sigma0
        S2_anti = self.S1
        S2_anti.requires_grad = True
        sigma_anti = self.sigma0
        for i in range(nSteps):
            W1 = asset_normals[1, :, i, :].reshape((-1))
            W2 = self.rho*W1 + np.sqrt(1-self.rho*self.rho)*vol_normals[1, :, i, :].reshape((-1))
            # Update S
            S2 = torch.maximum(S2 + sigma*torch.pow(S2, self.beta)*torch.sqrt(stepSize2)*W1, Tensor([0]))
            # Update sigma
            sigma = sigma*exp(-0.5*self.alpha*self.alpha*stepSize2 + self.alpha * np.sqrt(stepSize2)*W2)
            if antithetic:
                W1_anti = -asset_normals[1, :, i, :].reshape((-1))
                W2_anti = self.rho * W1_anti + np.sqrt(1 - self.rho * self.rho) * (-vol_normals[1, :, i, :].reshape((-1)))
                # Update S
                S2_anti = torch.maximum(S2 + sigma_anti * torch.pow(S2, self.beta) * torch.sqrt(stepSize2) * W1_anti, Tensor([0]))
                # Update sigma
                sigma_anti = sigma_anti * exp(-0.5 * self.alpha * self.alpha * stepSize2 + self.alpha * np.sqrt(stepSize2) * W2_anti)

        # Prepare output
        x = self.S1.detach().numpy().reshape((-1, 1))  # input spots
        y = self.payoff.evaluate(S2, self.K)  # payoff
        xbar = torch.autograd.grad(y, S2, grad_outputs=torch.ones_like(y))[0]  # differentials
        if antithetic:
            y_anti = self.payoff.evaluate(S2_anti, self.K)
            xbar_anti = torch.autograd.grad(y_anti, S2_anti, grad_outputs=torch.ones_like(y))[0]
            y = 0.5 * (y + y_anti)
            xbar = 0.5 * (xbar + xbar_anti)
        y = y.detach().numpy().reshape((-1, 1))
        xbar = xbar.detach().numpy().reshape(-1, 1)
        self.sim_time = round(time.time() - tmp, 3)
        return x, y, xbar

    def mc_estimate(self, S1_input, nPaths=10000, nSteps=100, seed=None):
        # Estimate prices and differentials with Monte-Carlo simulation
        if seed is None:
            np.random.seed(np.random.randint(1e6))
        else:
            np.random.seed(seed)

        asset_normals = torch.normal(0, 1, size=[len(S1_input), nPaths, nSteps, 1])
        vol_normals = torch.normal(0, 1, size=[len(S1_input), nPaths, nSteps, 1])
        stepSize = Tensor([(self.T2 - self.T1) / nSteps])

        y_mc = np.zeros_like(S1_input)
        y_sd = np.zeros_like(S1_input)
        xbar_mc = np.zeros_like(S1_input)
        xbar_sd = np.zeros_like(S1_input)
        for j, S1 in enumerate(S1_input):
            # Simulate exposure date paths
            S1 = Tensor(np.repeat(S1, nPaths))
            S1.requires_grad = True
            sigma = self.sigma0
            for i in range(nSteps):
                W1 = asset_normals[j, :, i, :].reshape((-1))
                W2 = self.rho * W1 + np.sqrt(1 - self.rho * self.rho) * vol_normals[j, :, i, :].reshape(-1)
                # Update S
                S1 = torch.maximum(S1 + sigma * torch.pow(S1, self.beta) * torch.sqrt(stepSize) * W1, Tensor([0]))
                # Update sigma
                sigma = sigma * exp(-0.5 * self.alpha * self.alpha * stepSize + self.alpha * np.sqrt(stepSize) * W2)
            S2 = S1
            # Evaluate payoffs
            y = self.payoff.evaluate(S2, self.K)
            # Evaluate differentials
            xbar = torch.autograd.grad(y, S2, grad_outputs=torch.ones_like(y))
            # Prepare output
            y = y.detach().numpy()
            y_mc[j] = y.mean()
            y_sd[j] = y.std()
            xbar = xbar[0].detach().numpy()
            xbar_mc[j] = xbar.mean()
            xbar_sd[j] = xbar.std()
        return y_mc, xbar_mc, y_sd, xbar_sd

    def test_set(self, lower=0.5, upper=1.5, num=10, nPaths=10000, nSteps=100, seed=1):
        spots = np.linspace(lower, upper, num)
        y_mc, xbar_mc, y_sd, xbar_sd = self.mc_estimate(spots, nPaths, nSteps, seed=seed)
        return spots, spots, y_mc.reshape((-1, 1)), xbar_mc.reshape((-1, 1))


def generate_multivar_structure(nStocks, basketSigma, w=None, corr=None, vols=None):
    # Prepare weights
    if w is None:
        w = np.random.uniform(1, 10, size=nStocks)
    w_out = w / np.sum(w)  # normalize

    # Prepare correlations
    if corr is None:
        randoms = np.random.uniform(low=-1, high=1, size=[2 * nStocks, nStocks])
        cov = np.matmul(randoms.T, randoms)
        invvols = np.diag(1. / np.sqrt(np.diagonal(cov)))
        corr = np.linalg.multi_dot([invvols, cov, invvols])
    corr_out = corr

    # Prepare volatilities
    if vols is None:
        vols = np.random.uniform(1, 10, size=nStocks)
    weightedVols = w_out * vols
    volNormalizer = np.sqrt(np.linalg.multi_dot([weightedVols.T, corr_out, weightedVols]))
    vols_out = vols * basketSigma / volNormalizer

    # Prepare covariance
    diagVols = np.diag(vols_out)
    cov_out = np.linalg.multi_dot([diagVols, corr_out, diagVols])
    return w_out, vols_out, corr_out, cov_out


class Bachelier(Generator):
    def __init__(self, basketSigma=0.2, nStocks=1, K=1.1, S0=1, T1=1, T2=2, sigmaAdj=1.5):
        self.basketSigma = basketSigma
        self.nStocks = nStocks
        self.K = K
        self.S0 = S0
        self.T1 = T1
        self.T2 = T2
        self.sigmaAdj = sigmaAdj

    def sim_paths(self, nPaths, w=None, corr=None, vols=None, seed=None, antithetic=True):
        tmp = time.time()
        if seed is None:
            np.random.seed(np.random.randint(1e6))
        else:
            np.random.seed(seed)

        # Generate weights, correlations etc.
        self.w, self.vols, self.corr, self.cov = generate_multivar_structure(self.nStocks,
                                                                             self.basketSigma,
                                                                             w=w,
                                                                             corr=corr,
                                                                             vols=vols)
        self.chol1 = np.linalg.cholesky(self.cov) * self.sigmaAdj * np.sqrt(self.T1)
        self.chol2 = self.chol1 / self.sigmaAdj * np.sqrt((self.T2 - self.T1) / self.T1)

        # Simulate paths of underlying
        normals = normal(size=[2, nPaths, self.nStocks])
        S0vec = np.repeat(self.S0, self.nStocks) # assuming same t=0 price for all stocks in the basket, WLOG.
        vol1 = np.matmul(normals[0, :, :], self.chol1.T)
        vol2 = np.matmul(normals[1, :, :], self.chol2.T)
        S1 = S0vec + vol1
        S2 = S1 + vol2
        self.S2 = S2

        # Evaluate time t=1 value of time t=2 call payoff
        basket2 = np.matmul(S2, self.w)
        payoff = np.maximum(basket2 - self.K, 0)

        # Compute pathwise differentials
        deltas = np.where(basket2 > self.K, 1, 0).reshape((-1, 1)) * self.w.reshape((1, -1))

        # Optional antithetic sampling. Default is true
        if antithetic:
            S2_anti = S1 - vol2
            basket2_anti = np.matmul(S2_anti, self.w)
            payoff_anti = np.maximum(basket2_anti - self.K, 0)
            deltas_anti = np.where(basket2_anti > self.K, 1, 0).reshape((-1, 1)) * self.w.reshape((1, -1))

            # Prepare output
            x = S1
            y = 0.5*(payoff + payoff_anti)
            xbar = 0.5*(deltas + deltas_anti)
        else:
            # Prepare output
            x = S1
            y = payoff
            xbar = deltas
        self.sim_time = round(time.time() - tmp, 3)
        return x, y, xbar

    def true_price(self, S1_input):
        # The bachelier formula as presented in Terakado (2019)
        vol = self.basketSigma * np.sqrt(self.T2 - self.T1)
        z = (S1_input - self.K)/vol
        return (S1_input - self.K)*norm.cdf(z) + vol*norm.pdf(z)

    def true_delta(self, S1_input):
        vol = self.basketSigma * np.sqrt(self.T2 - self.T1)
        z = (S1_input - self.K)/vol
        return norm.cdf(z)

    def test_set(self, lower=0.5, upper=1.5, num=2**11, seed=1):
        np.random.seed(seed)
        baskets = np.linspace(lower, upper, num)
        spots = np.random.uniform(lower, upper, size=(num, self.nStocks))
        # Scale spots such that they sum to the desired basket values
        adj = baskets / np.matmul(spots, self.w)
        spots = (spots.T * adj).T
        # Get analytics
        true_y = self.true_price(baskets)
        true_delta = np.matmul(self.true_delta(baskets).reshape(-1, 1), self.w.reshape((1, -1)))
        return spots, baskets.reshape((-1, 1)), true_y.reshape((-1, 1)), true_delta
