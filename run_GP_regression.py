import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

import torch
import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import Kernel, ScaleKernel, LinearKernel, RBFKernel
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

torch.set_default_dtype(torch.float64)
gpytorch.settings.cholesky_max_tries._set_value(100)
gpytorch.settings.lazily_evaluate_kernels._set_state(False)


class DistKernel(Kernel):
    has_lengthscale = True

    def __init__(self, dists, lengthscale_prior = None):
        super().__init__(lengthscale_prior = lengthscale_prior)
        self.dists = dists

    def forward(self, x1_index, x2_index, diag=False, **params):
        x1_index_ = x1_index.squeeze().to(torch.int32)
        x2_index_ = x2_index.squeeze().to(torch.int32)

        if x1_index_.dim() == 1:
            dist = self.dists[x1_index_, :][:, x2_index_]
        else:
            dist = torch.zeros(x1_index_.shape[0], x1_index_.shape[1], x2_index_.shape[1])
            for i in range(x1_index_.shape[0]):
                dist[i] = self.dists[x1_index_[i], :][:, x2_index_[i]]

        dist = dist.div(self.lengthscale)
        k = dist.pow(2).div_(-2).exp_()
        if diag:
            k = k[0]
        return k

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, dists1 = None, dists2 = None, dists3 = None, dists4 = None):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ZeroMean()
        if kernel == "Linear":
            self.covar_module = LinearKernel()
        elif kernel == "RBF":
            self.covar_module = ScaleKernel(RBFKernel())
        elif kernel == "SK":
            self.covar_module = ScaleKernel(DistKernel(dists1))
        elif kernel == "MK":
            covar_module1 = ScaleKernel(DistKernel(dists1))
            covar_module2 = ScaleKernel(DistKernel(dists2))
            covar_module3 = ScaleKernel(DistKernel(dists3))
            covar_module4 = ScaleKernel(DistKernel(dists4))
            self.covar_module = covar_module1 + covar_module2 + covar_module3 + covar_module4

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def main(args):
    X = torch.tensor(np.load(os.path.join(args.data_dir, "X.npy")), dtype = torch.float64)
    y = torch.tensor(np.load(os.path.join(args.data_dir, "y.npy")), dtype = torch.float64)
    sample_index = torch.tensor(list(range(len(y))))
    train_x, test_x, train_y, test_y, train_sample_index, test_sample_index = train_test_split(X, y, sample_index, test_size = 0.5, random_state = args.seed, shuffle = True)

    y_mean = train_y.mean()
    y_std = train_y.std()
    train_y = (train_y - y_mean) / y_std
    test_y = (test_y - y_mean) / y_std

    likelihood = GaussianLikelihood(noise_constraint = gpytorch.constraints.Positive())

    if args.kernel == "Linear" or args.kernel == "RBF":
        model = ExactGPModel(train_x, train_y, likelihood, args.kernel)
    elif args.kernel == "SK":
        train_x = train_sample_index
        test_x = test_sample_index
        dists1 = torch.tensor(np.load(os.path.join(args.data_dir, args.dist)))
        model = ExactGPModel(train_x, train_y, likelihood, args.kernel, dists1)
    elif args.kernel == "MK":
        train_x = train_sample_index
        test_x = test_sample_index
        dists1 = torch.tensor(np.load(os.path.join(args.data_dir, "dist_ja.npy")))
        dists2 = torch.tensor(np.load(os.path.join(args.data_dir, "dist_bc.npy")))
        dists3 = torch.tensor(np.load(os.path.join(args.data_dir, "dist_uu.npy")))
        dists4 = torch.tensor(np.load(os.path.join(args.data_dir, "dist_wu.npy")))
        model = ExactGPModel(train_x, train_y, likelihood, args.kernel, dists1, dists2, dists3, dists4)

    training_iter = 200
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    loss_hist = []
    best_loss = float("inf")
    for i in tqdm(range(training_iter)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.item())
        if best_loss > loss_hist[-1]:
            best_loss = loss_hist[-1]
            best_model = model.state_dict()

    if args.kernel == "SK":
        path = f"model_GP_SK_{os.path.basename(args.dist).split('.')[0]}_{args.seed}.pt"
    else:
        path = f"model_GP_{args.kernel}_{args.seed}.pt"
    torch.save(best_model, f"{args.result_dir}/{path}")
    model.load_state_dict(torch.load(f"{args.result_dir}/{path}"))

    result = {}
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = likelihood(model(train_x))

    output_mean = output.mean.detach() * y_std + y_mean
    output_variance = output.variance.detach() * y_std ** 2
    train_y_ = y_mean + train_y * y_std
    result["train_y_true"] = train_y_.tolist()
    result["train_y_pred"] = output_mean.tolist()
    result["train_y_variance"] = output_variance.tolist()

    log_likelihood = norm.logpdf(train_y_, output_mean, output_variance.sqrt())
    result["train_nll"] = -log_likelihood.mean().item()
    result["train_mse"] = mean_squared_error(train_y_, output_mean).item()
    result["train_rmse"] = root_mean_squared_error(train_y_, output_mean).item()
    result["train_r2"] = r2_score(train_y_, output_mean).item()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = likelihood(model(test_x))

    output_mean = output.mean.detach() * y_std + y_mean
    output_variance = output.variance.detach() * y_std ** 2
    test_y_ = y_mean + test_y * y_std
    result["test_y_true"] = test_y_.tolist()
    result["test_y_pred"] = output_mean.tolist()
    result["test_y_variance"] = output_variance.tolist()

    log_likelihood = norm.logpdf(test_y_, output_mean, output_variance.sqrt())
    result["test_nll"] = -log_likelihood.mean().item()
    result["test_mse"] = mean_squared_error(test_y_, output_mean).item()
    result["test_rmse"] = root_mean_squared_error(test_y_, output_mean).item()
    result["test_r2"] = r2_score(test_y_, output_mean).item()

    if args.kernel == "SK":
        path = f"result_GP_SK_{os.path.basename(args.dist).split('.')[0]}_{args.seed}.json"
    else:
        path = f"result_GP_{args.kernel}_{args.seed}.json"
    with open(f"{args.result_dir}/{path}", "w") as f:
        json.dump(result, f, indent = 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str)
    parser.add_argument("--result_dir", type = str)
    parser.add_argument("--seed", default = 0, type = int)
    parser.add_argument("--kernel", choices = ["Linear", "RBF", "SK", "MK"], type = str)
    parser.add_argument("--dist", default = "dist.npy", type = str)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.result_dir, exist_ok = True)
    main(args)
