import numpy as np
import pandas as pd
from scipy import spatial, stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.robust as robust
from statsmodels.stats import multitest
import seaborn as sns


################
#  PARAMETERS  #
################
sns.set_style('ticks')

chc_color = (0.894, 0.100, 0.107)
non_color = (0.441, 0.446, 0.446)
ais_color = (0.616, 0.07, 0)

syn_per_conn_color = (0, 0.651, 0.316)
num_conn_color = (0.965, 0.543, 0.122)

num_pot_color = (0.713, 0.14, 0.404)
conn_frac_color = (0.337, 0.702, 0.702)

shallow_color = (0.283, 0.471, 0.817)   # from muted option of sns default pallette
deep_color = (0.005, 0.111, 0.499)      # from deep option of sns default pallette 
stars = [0.05, 0.01, 0.001]

plot_label_lookup = {'syn_net_chc': '# ChC Syn.',
                     'syn_net_non': '# Non-ChC Syn.',
                     'soma_y_adj': 'Soma Depth ($\mu m$)',
                     'soma_y_um': 'Soma Depth ($\mu m$)',
                     'soma_x_um': 'Soma Mediolateral Pos. ($\mu m$)',
                     'n_syn_soma': '# Syn Soma (cutout)',
                     'soma_synapses': '# Syn Soma',
                     'soma_area': 'Soma Area ($\mu m^2$)',
                     'soma_syn_density': '# Syn Soma/($\mu m^2$)',
                     'num_cells_chc': '# ChC Connections',
                     'syn_mean_chc': '# Syn/Connection',
                     'conn_frac': 'Connectivity Fraction',
                     'num_potential': '# Potential ChC',
                     'size_mean_chc': 'Mean ChC Syn Size',
                    }

tick_dict = {'syn_net_chc': np.arange(0,27,5),
             'syn_net_non': np.arange(0,27,5),
             'soma_y_adj': np.arange(0, 121, 20),
             'soma_x_um': np.arange(0, 251, 50),
             'n_syn_soma': np.arange(60,161,20),
             'soma_synapses': np.arange(40, 121, 20),
             'soma_area': np.arange(500, 901, 100),
             'soma_syn_density': np.arange(0.05, 0.15, 0.02),
             'num_cells_chc': np.arange(0,10,2.5),
             'syn_mean_chc': np.arange(0,8,2),
             'conn_frac': np.arange(0,1.01,0.2),
             'num_potential': np.arange(0, 26, 5),
             }


axis_label_font = {'family': 'sans-serif',
                   'weight': 'normal',
                   'size': 12,
                   }

axis_tick_font = {'family': 'sans-serif',
                   'weight': 'normal',
                   'size': 10,
                   }

################################
#  PROPERTY SETTING FUNCTIONS  #
################################

def set_rc_params(mpl):
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Helvetica Neue']

def format_axis(ax, axis_info):
    ax.set_xlabel(axis_info['xlabel'])
    ax.set_ylabel(axis_info['ylabel'])
    ax.set_xlim(axis_info.get('xstart',-0.5), axis_info['xmax'])
    sns.despine(offset=5, trim=True, ax=ax)
    
def set_axis_fonts(ax, label_font=axis_label_font, tick_font=axis_tick_font, ytick_int=False, xtick_int=False, xprecision=1, yprecision=1):
    ax.set_xlabel(ax.get_xlabel(), fontdict=label_font)
    ax.set_ylabel(ax.get_ylabel(), fontdict=label_font)

    if ytick_int:
        yformat_str = '{:d}'
        ax.set_yticklabels([yformat_str.format(int(y)) for y in ax.get_yticks()],
                       fontdict=tick_font)        
    else:
        yformat_str = '{:.{yprecision}f}'
        ax.set_yticklabels([yformat_str.format(y, yprecision=yprecision) for y in ax.get_yticks()],
                       fontdict=tick_font)

    if xtick_int:
        xformat_str = '{:d}'
        ax.set_xticklabels([xformat_str.format(int(x)) for x in ax.get_xticks()],
                   fontdict=tick_font)

    else:
        xformat_str = '{:.{xprecision}f}'
        ax.set_xticklabels([xformat_str.format(x, xprecision=xprecision) for x in ax.get_xticks()],
                       fontdict=tick_font)
        
def set_axis_size(w,h, ax):
    """ w, h: width, height in inches """
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


#########################
#  ACCESSORY FUNCTIONS  #
#########################


def assign_stars(pvals, star_ths):
    n_stars = np.zeros(len(pvals))
    for ii, star in enumerate(star_ths):
        n_stars[pvals < star] = ii+1
    return n_stars


def plot_stars(xs, ys, n_stars, ax, xytext=(5,0), fontsize=12, fontweight=100, color=None, horizontalalignment='left'):
    for x, y, ns in zip(xs, ys, n_stars):
        if ns>0:
            ax.annotate('*'*int(ns),
                         (x,y),
                         textcoords='offset points',
                         xytext=xytext,
                         fontsize=fontsize,
                         fontweight=fontweight,
                         color=color,
                         horizontalalignment=horizontalalignment)


def jitter(pts, jit=0.1):
    return jit * 2*(np.random.rand(len(pts))-0.5)


def convex_hull_xy( pts ):
    hull = spatial.ConvexHull(pts)
    xpts = np.concatenate((pts[hull.vertices,0], pts[hull.vertices[[-1,0]],0]))
    ypts = np.concatenate((pts[hull.vertices,1], pts[hull.vertices[[-1,0]],1]))
    return xpts, ypts


####################
#  BASIC PLOTTING  #
####################


def barplot_data(data, color, xmax, bins, figsize, axis_info, figax=(None, None)):
    fig, ax = figax
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
    fig.set_facecolor((1,1,1))        
    sns.distplot(data,
                 bins=bins,
                 color=color,
                 ax=ax,
                 kde=False,
                 hist_kws={'alpha': 1, 'edgecolor':'w'})
    format_axis(ax, axis_info)
    return fig, ax


def make_scatterplot(x, y, df, figsize, label_lookup, tick_dict,
                     xtick_int=True, ytick_int=True, xprecision=1,
                     yprecision=1, xlim=None, ylim=None, color='k'):
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(x=x, y=y,
                         data=df,
                         color=color,
                         s=25,
                         alpha=0.8,
                         ax=ax)
    _=ax.set_xlabel(label_lookup[x])
    _=ax.set_ylabel(label_lookup[y])
#     ax.grid(True, axis='both')
    ax.grid(False)
    ax.set_axisbelow(True)
    sns.despine(offset=2, trim=False, ax=ax)

    ax.set_xticks(tick_dict[x])
    ax.set_yticks(tick_dict[y])
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    set_axis_fonts(ax, ytick_int=ytick_int, xtick_int=xtick_int, xprecision=xprecision, yprecision=yprecision)
    set_axis_size(figsize[0], figsize[1], ax)
    return fig, ax


def plot_data_pair(data, column, color_chc, color_non, xlabel=None, xmax_extra=2,
                   **kwargs):
    column_chc = column+'_chc'
    xlabel_chc = xlabel + ' ChC'
    
    column_non = column + '_non'
    xlabel_non = xlabel + ' Non-ChC'
    xmax = max( max(data[column_chc]), max(data[column_non])) + xmax_extra
    fig_chc, ax_chc = plot_data_single(data, column_chc, color_chc, xlabel_chc, xmax=xmax, **kwargs)
    fig_non, ax_non = plot_data_single(data, column_non, color_non, xlabel_non, xmax=xmax, **kwargs)
    return fig_chc, ax_chc, fig_non, ax_non


def plot_data_single(data, column, color, xlabel, figsize=(4,4), bin_start=0, bin_increment=1,
                     xmax=None, xmax_extra=2, axis_label_font={}, axis_tick_font={},
                     xtick_int=True, ytick_int=True, ylabel='#AIS'):
    datapts = data[column]
    if xmax is None:
        xmax = max(datapts) + xmax_extra
    bins = np.arange(bin_start, xmax, bin_increment)-0.5

    if xlabel is None:
        xlabel = column

    axis_info = {'title': column,
                 'xlabel': xlabel,
                 'ylabel': ylabel,
                 'xmax': xmax}
    
    fig, ax = barplot_data(datapts, color, xmax, bins, figsize, axis_info)
    set_axis_size(*figsize, ax)
    set_axis_fonts(ax, axis_label_font, axis_tick_font, xtick_int=xtick_int, ytick_int=ytick_int)
    return fig, ax



###################
#  OLS FUNCTIONS  #
###################


def ols_analysis_single(df, row_filter, ycol, xcols, color, plot_label_lookup=plot_label_lookup, stars=stars, robust=True, xticks=None):
    if robust:
        res = fit_rlm(xcols, ycol, row_filter, df)
    else:
        res = fit_ols(xcols, ycol, row_filter, df)
    
    column_names = [plot_label_lookup[col] for col in xcols]
    var_name = plot_label_lookup[ycol]
    
    fig, ax = plot_ols_fit_single(res, robust, column_names, color, stars, var_name, xticks=xticks)
    res_df = ols_results_dataframe([res], column_names, [plot_label_lookup[ycol]])
    return fig, ax, res_df, res


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

def compute_residuals(columns_x, column_y, rowfilter, df, use_robust=True):
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
        if use_robust:
            mod = sm.RLM(yz, Xz[:, relcols], M=robust.norms.HuberT())
        else:
            mod = sm.OLS(yz, Xz[:, relcols])
        res=mod.fit()
        col_residuals.append([Xz[:,col]*col_stds[col]+col_means[col], res.resid * y_std])
    return col_residuals

def residual_scatterplots(y_col, columns, row_filter, df, prefix, plot_dir, plot_label_lookup=plot_label_lookup, robust=True):
    resids = compute_residuals(columns, y_col, row_filter, df, use_robust=robust)
    for col_ind in range(len(columns)):
        fig, ax = plt.subplots(figsize=(3,3))
        sns.regplot(x=resids[col_ind][0], y=resids[col_ind][1],
                    ax=ax, marker='o', color='k', scatter_kws={'s':6, 'color':(0, 0, 0)})
#         ax.plot(resids[col_ind][0], resids[col_ind][1], 'k.', alpha=0.8)
        ax.set_ylabel(f'{plot_label_lookup[y_col]} Residual')
        ax.set_xlabel(plot_label_lookup[columns[col_ind]])
        ax.set_title(f'r={stats.pearsonr(resids[col_ind][0], resids[col_ind][1])[0]:.2f}')
        sns.despine(ax=ax, offset=5)
        fig.savefig(f'{plot_dir}/residual_scatterplot_{prefix}_{y_col}_{columns[col_ind]}_{"rls" if robust else "ols"}.pdf', bbox_inches='tight')

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


def plot_ols_fit_single(res, robust, column_labels, color, stars, title, xticks=None, figsize=(3,1.5), title_pad=15):
    fig, ax = plt.subplots(figsize=figsize)
    
    if xticks is None:
        xticks = np.arange(-0.5,0.51,0.25)

    param_vals = res.params
    conf_ints = res.conf_int()
    xvals = np.arange(len(param_vals))
    for ii in np.arange(len(param_vals)):
        ax.plot(conf_ints[ii,:], [xvals[ii], xvals[ii]], linestyle='-', color=color)
    ax.plot(param_vals, xvals, marker='s', linestyle='', color=color, markeredgecolor='w')
        
    _, p_corr, _, _ = multitest.multipletests(res.pvalues)
    plot_stars(param_vals, xvals, assign_stars(p_corr, stars), ax,
               xytext=(0,3), color='k', horizontalalignment='center')

    ax.set_title(title, pad=title_pad)
    ax.set_yticks( np.arange(len(param_vals)))
    _=ax.set_yticklabels(column_labels)
    ax.set_ylim([-0.2, len(param_vals)-1+0.2])
    
    ax.plot([0,0], [-0.5,len(param_vals)-1+0.8], 'k', zorder=1, linewidth=2)
    plt.grid(axis='x', which='major')
    ax.set_axisbelow(True)
    
    ax.set_xticks(xticks)
    if not robust:
        ax.set_xlabel('OLS Regression Coef.')
    else:
        ax.set_xlabel('RLS Regression Coef.')

    sns.despine(ax=ax, offset=5, trim=True)

    return fig, ax



def plot_ols_fit(results, column_labels, colors, stars, legend_names, figsize=(4,3), offset=0.3):
    fig, ax = plt.subplots(figsize=figsize)
    xoffset = 0
    for ind, res in enumerate(results):
        param_vals = res.params
        conf_ints = res.conf_int()
        xvals = np.arange(len(param_vals)) + xoffset
        for ii in np.arange(len(param_vals)):
            ax.plot([xvals[ii], xvals[ii]], conf_ints[ii,:], linestyle='-', color=colors[ind])
        ax.plot(xvals, param_vals, marker='s', linestyle='', color=colors[ind])
        xoffset+=offset
        
        _, p_corr, _, _ = multitest.multipletests(res.pvalues)
        plot_stars(xvals, conf_ints[:,1], assign_stars(p_corr, stars), ax,
                   xytext=(0,0), color='k', horizontalalignment='center')

    lines = [plt.Line2D([0],[0], linestyle='-', color=color) for color in colors]
    ax.legend(lines, legend_names)
    
    ax.set_xticks( np.arange(len(param_vals))+offset/2)
    sns.despine(ax=ax, offset=5, trim=True)
    
    ax.plot([-0.5,len(param_vals)-1+0.8],[0,0], 'k', zorder=1, linewidth=2)
    plt.grid(axis='y', which='major')
    ax.set_axisbelow(True)
    
    _=ax.set_xticklabels(column_labels, rotation=45)

    ax.set_ylabel('OLS Regression Coef.')
    ax.set_xlim([-offset/2, len(param_vals)-1+1.5*offset])
    return fig, ax


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




