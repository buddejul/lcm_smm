# Simulated methods of moments for LCM topics class
# import estimagic
from numpy.random import default_rng
import numpy as np
import plotly.express as px
import pandas as pd
from pathlib import Path

import estimagic as em

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True
pd.options.plotting.backend = "plotly"

this_dir = Path(".")
this_dir = this_dir.resolve()
root = this_dir
print(root)
bld = root / "bld"
src = root / "src"

bld.mkdir(parents=True, exist_ok=True)

def smm(reps, data, initial_value): 

    obs = len(data["Y"])

    observed_moments = []
    for var in ["A", "B", "C"]:
        observed_moments.append(_compute_cross_moment(data["Y"], data[var]))

    errors = _draw_errors(reps, obs)

    def criterion(beta):
        """Compute SMM criterion function as fct of beta only for optimizer."""
        return _compute_smm_criterion(beta, data, observed_moments, errors, reps)

    res = em.minimize(
        criterion=criterion,
        params=initial_value,
        algorithm="scipy_lbfgsb",
    )

    return res

def _draw_errors(reps, obs):
    """Draw all errors (reps x obs) for SMM."""
    rng = default_rng()
    e = rng.normal(0, 1, (reps, obs))
    return e

def _compute_y(X, e, beta):
    """Compute y given X, e, and beta."""
    return X @ beta + e

def _square_loss(target, estimate):
    """Compute square loss for a vector of targets and an estimate."""
    return np.sum((target - estimate) ** 2)

def _compute_cross_moment(x, y):
    """Compute cross-moment for two vectors."""
    return np.sum(x * y) / len(x)

def _compute_smm_criterion(beta, data, observed_moments, e, reps):
    """Compute SMM criterion function."""
    simulated_moments = np.zeros((reps, len(beta)))

    X = data.drop("Y", axis=1)

    for rep in range(reps):
        y = _compute_y(X, e[rep,:], beta)
        simulated_moments[rep, 0] = _compute_cross_moment(y, data["A"])
        simulated_moments[rep, 1] = _compute_cross_moment(y, data["B"])
        simulated_moments[rep, 2] = _compute_cross_moment(y, data["C"])

    average_sim_moments = np.mean(simulated_moments, axis=0)

    criterion = _square_loss(observed_moments, average_sim_moments)
    return criterion

# Script to run SMM
if __name__ == "__main__":

    np.random.seed(123)
    reps = 1000
    initial_value = np.array([1, 1, 1])
    data = pd.read_feather(src / "data.feather")
    res = smm(reps, data, initial_value)
    print(res)
