# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:00:51 2025

@author: ichel
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh

def eval_grid_baseline(ATA, ATy, A_test, y_test, mus=None):
    """
    Compute validation loss C(mu) and condition number cond(ATA+muI) over a grid.

    Parameters
    ----------
    ATA : (d,d) ndarray
        A^T A from training data
    ATy : (d,) ndarray
        A^T y from training data
    A_test : (n_test,d) ndarray
        Test matrix
    y_test : (n_test,) ndarray
        Test output
    mus : array_like, optional
        Grid of mu values to evaluate. Default: logspace(-12, 6, 200)

    Returns
    -------
    mus : ndarray
        Evaluated mu grid
    C_vals : ndarray
        Validation loss for each mu
    conds : ndarray
        Condition number of ATA+muI for each mu
    """
    if mus is None:
        mus = np.logspace(-12, 6, 200)

    C_vals = []
    conds = []
    d = ATA.shape[0]
    I = np.eye(d)

    for mu in mus:
        K = ATA + mu * I
        # Solve (K x = ATy)
        x = np.linalg.solve(K, ATy)
        r = A_test @ x - y_test
        C = 0.5 * float(r @ r)
        C_vals.append(C)
        # Condition number
        eigs = eigvalsh(K)
        conds.append(eigs.max() / (eigs.min() + 1e-30))

    return mus, np.array(C_vals), np.array(conds)


def plot_grid_baseline(mus, C_vals, conds, mu_star=None, title=None):
    fig, ax1 = plt.subplots(figsize=(6,4))

    # Plot validation loss
    ax1.loglog(mus, C_vals, label="Validation loss C(mu)")
    ax1.set_xlabel("mu (log scale)")
    ax1.set_ylabel("C(mu)")

    # Plot condition number on secondary axis
    ax2 = ax1.twinx()
    ax2.loglog(mus, conds, ls="--", color="orange", label="cond(ATA+muI)")
    ax2.set_ylabel("cond(K)")

    if mu_star is not None:
        ax1.axvline(mu_star, ls=":", color="red", label=f"chosen mu={mu_star:.2e}")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    if title: fig.suptitle(title)
    plt.tight_layout()
    plt.show()




[A, A_test, y, y_test, ATA, ATy, AAT] = pkl.load(open('file.pkl', 'rb'))


# Suppose you have A, y, A_test, y_test
ATA = A.T @ A
ATy = A.T @ y

mus, C_vals, conds = eval_grid_baseline(ATA, ATy, A_test, y_test)

plot_grid_baseline(mus, C_vals, conds, title="Grid baseline (oracle)")


