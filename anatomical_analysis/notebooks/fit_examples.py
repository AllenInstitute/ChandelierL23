import statsmodels.api as sm
import statsmodels.robust as robust
from statsmodels.stats import multitest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fit_ols(columns_x, column_y, rowfilter, df):
    X = df[rowfilter][columns_x].values.astype(float)
    X[np.isinf(X)] = np.nan
    Xz = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)

    y = df[rowfilter][column_y].values.astype(float)
    y[np.isinf(y)] = np.nan
    yz = (y-np.nanmean(y)) / np.nanstd(y)
    good_row = ~np.any(np.isnan(np.hstack((yz.reshape(-1,1), Xz))), axis=1)

    mod = sm.OLS(yz[good_row], Xz[good_row])
    return mod.fit()


def fit_rlm(columns_x, column_y, rowfilter, df):
    X = df[rowfilter][columns_x].values.astype(float)
    X[np.isinf(X)] = np.nan
    Xz = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)

    y = df[rowfilter][column_y].values.astype(float)
    y[np.isinf(y)] = np.nan
    yz = (y-np.nanmean(y)) / np.nanstd(y)
    
    good_row = ~np.any(np.isnan(np.hstack((yz.reshape(-1,1), Xz))), axis=1)
    
    mod = sm.RLM(yz[good_row], Xz[good_row], M=robust.norms.HuberT())
    return mod.fit()


def compute_residuals(columns_x, column_y, rowfilter, df, robust=True):
    """For each column in columns_x, do regression on everything else
    and get the residual values of column_y to plot against it.
    """

    X = df[rowfilter][columns_x].values.astype(float)
    X[np.isinf(X)] = np.nan
    Xz = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    
    col_means = np.nanmean(X, axis=0)
    col_stds = np.nanstd(X, axis=0)
    
    y = df[rowfilter][column_y].values.astype(float)
    y[np.isinf(y)] = np.nan
    yz = (y-np.nanmean(y)) / np.nanstd(y)
    
    y_mean = np.nanmean(y)
    y_std = np.nanstd(y)

    good_row = ~np.any(np.isnan(np.hstack((yz.reshape(-1,1), Xz))), axis=1)

    yz = yz[good_row]
    Xz = Xz[good_row]
    cols = np.arange(Xz.shape[1])
    col_residuals = []
    for col in cols:
        relcols = cols[cols!=col]
        if robust:
            mod = sm.RLM(yz, Xz[:,relcols], M=robust.norms.HuberT())
        else:
            mod = sm.OLS(yz[good_row], Xz[good_row])
        res=mod.fit()
        col_residuals.append([Xz[:,col]*col_stds[col]+col_means[col], res.resid * y_std])
    return col_residuals


def ols_results_dataframe(results, column_labels, names):
    df_dict = {'Variable': column_labels}
    for res, name in zip(results, names):
        df_dict['{} corr'.format(name)] = res.params
        _, p_corr, _, _ = multitest.multipletests(res.pvalues)
        df_dict['{} pval'.format(name)] = p_corr
    
    try:
        for res, name in zip(results, names):
            df_dict['{} R2'.format(name)] = [res_chc.rsquared,] + (len(res.params)-1) * [np.nan]
    except:
        pass
    return pd.DataFrame(df_dict)


def assign_stars(pvals, star_ths):
    n_stars = np.zeros(len(pvals))
    for ii, star in enumerate(star_ths):
        n_stars[pvals < star] = ii+1
    return n_stars


def plot_stars(xs, ys, n_stars, ax, xytext=(5, 0), fontsize=12, fontweight=100, color=None, horizontalalignment='left'):
    for x, y, ns in zip(xs, ys, n_stars):
        if ns > 0:
            ax.annotate('*'*int(ns),
                        (x, y),
                        textcoords='offset points',
                        xytext=xytext,
                        fontsize=fontsize,
                        fontweight=fontweight,
                        color=color,
                        horizontalalignment=horizontalalignment)


def plot_ols_fit_single(res, robust, column_labels, color, stars, title, xticks=None, figsize=(3, 1.5), title_pad=15):
    fig, ax = plt.subplots(figsize=figsize)

    if xticks is None:
        xticks = np.arange(-0.5, 0.51, 0.25)

    param_vals = res.params
    conf_ints = res.conf_int()
    xvals = np.arange(len(param_vals))
    for ii in np.arange(len(param_vals)):
        ax.plot(conf_ints[ii, :], [xvals[ii], xvals[ii]], linestyle='-', color=color)
    ax.plot(param_vals, xvals, marker='s', linestyle='', color=color, markeredgecolor='w')

    _, p_corr, _, _ = multitest.multipletests(res.pvalues)
    plot_stars(param_vals, xvals, assign_stars(p_corr, stars), ax,
               xytext=(0, 3), color='k', horizontalalignment='center')

    ax.set_title(title, pad=title_pad)
    ax.set_yticks(np.arange(len(param_vals)))
    _ = ax.set_yticklabels(column_labels)
    ax.set_ylim([-0.2, len(param_vals)-1+0.2])

    ax.plot([0, 0], [-0.5, len(param_vals)-1+0.8], 'k', zorder=1, linewidth=2)
    plt.grid(axis='x', which='major')
    ax.set_axisbelow(True)

    ax.set_xticks(xticks)
    if not robust:
        ax.set_xlabel('OLS Regression Coef.')
    else:
        ax.set_xlabel('RLS Regression Coef.')

    sns.despine(ax=ax, offset=5, trim=True)

    return fig, ax
