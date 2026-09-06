TOL = 0.000000001


def normalize_zscore(X, mean=None, std=None, tol=TOL):
    if mean is None or std is None:
        std = X.std(dim=0) + tol
        mean = X.mean(dim=0)

    normalized = (X - mean).div(std)
    return normalized, mean, std
