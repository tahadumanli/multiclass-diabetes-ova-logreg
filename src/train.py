

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict

CSV_PATH = "data/Multiclass Diabetes Dataset.csv"
CLASS_LABELS = ["non-diabet","predict-diabet","diabet"]

# -----------------------------
# 0) Utilities
# -----------------------------
def add_intercept(X: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    new = np.ones((m,1))
    return np.concatenate([new,X],axis=1)
    """
    TODO:
    - Given X (m x n), return X_augmented (m x (n+1)) with a leading column of 1s.
    """
# -----------------------------
# 1) Data loading and preprocessing
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def assert_no_missing(df: pd.DataFrame) -> None:

    if df.isna().any().any():
        raise ValueError("Missing values detected. Clean or impute before training.")
    


def build_xy(df: pd.DataFrame, target_col: str = "Class") -> Tuple[np.ndarray, np.ndarray]:
    X = df.drop(columns=[target_col]).to_numpy(dtype=float)
    y = df[target_col].astype(int).to_numpy()
    return X , y
    """
    TODO:
    - X = all columns except target_col, as float numpy array (m x n)
    - y_num = df[target_col].astype(int).to_numpy()
    - Return X, y_num
    """

def feature_normalize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0,ddof=0)
    sigma_safe=sigma.copy()
    sigma_safe[sigma_safe == 0] = 1.0
    X_norm = (X- mu) / sigma_safe
    return X_norm, mu, sigma
    """
    TODO:
    - Compute mean (mu) and std (sigma) per feature.
    - Return X_norm = (X - mu) / sigma, along with mu and sigma.
    - Important: if sigma[j] == 0, set sigma[j] = 1 to avoid divide-by-zero.
    """

def train_test_split_basic(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(m)
    rng.shuffle(idx)
    m_test = int(np.round(test_size * m))
    test_idx = idx[:m_test]
    train_idx = idx[m_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    """
    TODO:
    - Shuffle indices with the given seed.
    - Split into train and test by proportion test_size.
    - Later, model will be tested by test_size to measure how well does it work.
    - Return X_train, X_test, y_train, y_test
    """


# -----------------------------
# 2) Logistic regression core (binary)
# -----------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped=np.clip(z, -500, 500)
    return 1.0/ (1.0 + np.exp(-z_clipped))
    """
    TODO: Numerically stable sigmoid. Return 1 / (1 + exp(-z)).
    - For stability, clip z within a reasonable range, e.g., [-500, 500].
    """

def lr_cost_function(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float) -> float:
    m = X.shape[0]
    X_aug = add_intercept(X)
    h = sigmoid(X_aug @ theta)
    eps = 1e-12
    J_unreg = -(1.0/m) * (y @ np.log(h+eps)+(1-y) @ np.log(1-(h+eps)))
    reg = (lambda_ / (2*m)) * (theta[1:] @ theta[1:])
    return float(J_unreg + reg)
    """
    TODO:
    - theta: (n+1,) parameter vector for hypothesis h_theta(x) = sigmoid(X_aug * theta)
    - X: (m x n) features WITHOUT intercept column
    - y: (m,) binary labels {0,1}
    - lambda_: L2 regularization strength
    Steps:
      * Build X_aug with intercept.
      * Compute h = sigmoid(X_aug @ theta)
      * Compute unregularized cost J = -(1/m) * [y^T log h + (1-y)^T log(1-h)]
      * Add regularization: (lambda_/(2m)) * sum(theta[1:]^2)
    - Return cost J (float)
    """


def lr_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    m = X.shape[0]
    X_aug = add_intercept(X)
    h = sigmoid(X_aug @ theta)
    grad = (1.0/m) * (X_aug.T @ (h-y))
    reg = np.concatenate([[0.0], (lambda_/m)*theta[1:]])
    return grad+reg
    """
    TODO:
    - Return gradient dJ/dtheta (shape (n+1,))
    - Regularize all but theta[0].
    """


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
    alpha: float,
    num_iters: int,
    lambda_: float
) -> Tuple[np.ndarray, np.ndarray]:
    theta = theta0.copy()
    J_hist = np.zeros(num_iters, dtype=float)
    for t in range(num_iters):
        grad = lr_gradient(theta, X ,y, lambda_)
        theta -= alpha * grad
        J_hist[t]=lr_cost_function(theta, X, y, lambda_)
    return theta, J_hist
    """
    TODO:
    - Run batch gradient descent for num_iters.
    - On each step: theta = theta - alpha * grad
    - Track cost history into J_hist (length num_iters)
    - Return final theta and J_hist
    """

def predict_proba_binary(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_aug = add_intercept(X)
    return sigmoid(X_aug @ theta)
    """
    TODO:
    - Return predicted probabilities sigmoid(X_aug @ theta) for each sample.
    """

def predict_binary(theta: np.ndarray, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba_binary(theta, X) >= threshold).astype(int)
    """
    TODO:
    - Return predicted class labels {0,1} using threshold.
    """

# -----------------------------
# 3) One-vs-all wrapper
# -----------------------------
def one_vs_all(
    X: np.ndarray, y_num: np.ndarray, num_labels: int, alpha: float, num_iters: int, lambda_: float
) -> np.ndarray:
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n+1), dtype=float)
    for k in range(num_labels):
        yk = (y_num == k).astype(int)
        theta0 = np.zeros(n + 1, dtype=float)
        theta_k, _ = gradient_descent(X, yk, theta0, alpha, num_iters, lambda_)
        all_theta[k, :] = theta_k
    return all_theta
    """
    TODO:
    - Train K = num_labels binary classifiers with one-vs-all.
      For class k, set y_k = 1 if y==k else 0.
    - Initialize theta_k = zeros(n+1). Optimize with gradient_descent.
    - Stack thetas into all_theta (K x (n+1)). Row k corresponds to class k.
    - Return all_theta.
    """

def predict_one_vs_all(all_theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_aug = add_intercept(X)
    P = sigmoid(X_aug @ all_theta.T)
    y_pred = np.argmax(P, axis=1)
    return y_pred
    """
    TODO:
    - Compute probabilities for each class: P_k = sigmoid(X_aug @ all_theta[k]^T).
    - Predict class = argmax_k P_k for each sample.
    - Return y_pred (m,)
    """

# -----------------------------
# 4) Evaluation helpers
# -----------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)
    """
    TODO: Return mean(y_true == y_pred).
    """

def confusion_matrix_simple(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> np.ndarray:
    cm = np.zeros((num_labels, num_labels),dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    return cm    
    """
    TODO:
    - Build a (num_labels x num_labels) confusion matrix where rows are true, cols are predicted.
    """

# -----------------------------
# 5) Main experiment
# -----------------------------
def main() -> None:
    # Load and preprocess
    df = load_data(CSV_PATH)
    assert_no_missing(df)
    X, y = build_xy(df, "Class")

    # Split
    X_train, X_test, y_train, y_test = train_test_split_basic(X, y, test_size=0.2, seed=42)

    # Normalize using train stats, apply to both train and test
    X_train_norm, mu, sigma = feature_normalize(X_train)
    # apply same mu, sigma to test
    sigma_safe = sigma.copy(); sigma_safe[sigma_safe == 0] = 1.0
    X_test_norm = (X_test - mu) / sigma_safe

    # Hyperparameters
    alpha = 0.1         # learning rate
    num_iters = 2000    # iterations
    lambda_ = 1.0       # L2 regularization

    # Train one-vs-all
    num_labels = 3
    all_theta = one_vs_all(X_train_norm, y_train, num_labels, alpha, num_iters, lambda_)

    # Predict
    y_pred = predict_one_vs_all(all_theta, X_test_norm)

    # Evaluate
    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix_simple(y_test, y_pred, num_labels)

    print("Accuracy:", round(float(acc), 4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=CLASS_LABELS, columns=CLASS_LABELS))


if __name__ == "__main__":
    main()
