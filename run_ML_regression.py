import os
import json
import random
import argparse
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from ngboost import NGBRegressor


def main(args):
    result = {}

    X = np.load(os.path.join(args.data_dir, "X.npy"))
    y = np.load(os.path.join(args.data_dir, "y.npy"))
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.5, random_state = args.seed, shuffle = True)

    result["test_y"] = test_y.tolist()

    y_mean = train_y.mean()
    y_std = train_y.std()
    train_y = (train_y - y_mean) / y_std
    test_y = (test_y - y_mean) / y_std

    model = RandomForestRegressor(n_estimators = 100, n_jobs = -1, random_state = args.seed)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    test_y_ = y_mean + test_y * y_std
    pred_ = y_mean + pred * y_std
    result["rf_pred"] = pred_.tolist()
    result["rf_mse"] = mean_squared_error(test_y_, pred_).item()
    result["rf_rmse"] = root_mean_squared_error(test_y_, pred_).item()
    result["rf_r2"] = r2_score(test_y_, pred_).item()

    cv = KFold(n_splits = 5, shuffle = True, random_state = args.seed)
    model = RidgeCV(cv = cv)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    pred_ = y_mean + pred * y_std
    result["ridge_pred"] = pred_.tolist()
    result["ridge_mse"] = mean_squared_error(test_y_, pred_).item()
    result["ridge_rmse"] = root_mean_squared_error(test_y_, pred_).item()
    result["ridge_r2"] = r2_score(test_y_, pred_).item()

    model = LassoCV(cv = cv, max_iter = 5000, n_jobs = -1, random_state = args.seed)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    pred_ = y_mean + pred * y_std
    result["lasso_pred"] = pred_.tolist()
    result["lasso_mse"] = mean_squared_error(test_y_, pred_).item()
    result["lasso_rmse"] = root_mean_squared_error(test_y_, pred_).item()
    result["lasso_r2"] = r2_score(test_y_, pred_).item()

    model = ElasticNetCV(l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], cv = cv, max_iter = 5000, n_jobs = -1, random_state = args.seed)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    pred_ = y_mean + pred * y_std
    result["elasticnet_pred"] = pred_.tolist()
    result["elasticnet_mse"] = mean_squared_error(test_y_, pred_).item()
    result["elasticnet_rmse"] = root_mean_squared_error(test_y_, pred_).item()
    result["elasticnet_r2"] = r2_score(test_y_, pred_).item()

    ngb = NGBRegressor().fit(train_x, train_y)
    preds = ngb.pred_dist(test_x)
    output_mean = preds.loc * y_std + y_mean
    output_variance = preds.var * y_std ** 2
    test_y_ = y_mean + test_y * y_std

    result["ngb_pred"] = output_mean.tolist()
    result["ngb_variance"] = output_variance.tolist()
    log_likelihood = norm.logpdf(test_y_, output_mean, np.sqrt(output_variance))
    result["ngb_nll"] = -log_likelihood.mean().item()
    result["ngb_mse"] = mean_squared_error(test_y_, output_mean).item()
    result["ngb_rmse"] = root_mean_squared_error(test_y_, output_mean).item()
    result["ngb_r2"] = r2_score(test_y_, output_mean).item()

    with open(f"{args.result_dir}/result_ML_{args.seed}.json", "w") as f:
        json.dump(result, f, indent = 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str)
    parser.add_argument("--result_dir", type = str)
    parser.add_argument("--seed", default = 0, type = int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.result_dir, exist_ok = True)
    main(args)
