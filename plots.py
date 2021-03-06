import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import ptitprince as pt

# ----------
# Loss Plots
# ----------

def save_loss_plot(path, loss_function, v_path=None, show=True):
    df = pd.read_csv(path)
    if v_path is not None:
        vdf = pd.read_csv(v_path)
    else:
        vdf = None
    p = Path(path)
    n = p.stem
    d = p.parents[0]
    out_path = os.path.join(d, n + '_loss.png')
    fig, ax = plot_loss(df, vdf=vdf, x_lab='Iteration', y_lab=loss_function, save=out_path, show=show)



def plot_loss(df, vdf=None, x_lab='Iteration', y_lab='BCE Loss', save=None, show=True):
    x = df['Unnamed: 0'].values
    y = df['loss'].values
    epochs = len(df['epoch'].unique())
    no_batches = int(len(x) / epochs)
    epoch_ends = np.array([((i + 1) * no_batches) - 1 for i in range(epochs)])
    epoch_end_x = x[epoch_ends]
    epoch_end_y = y[epoch_ends]
    fig, ax = plt.subplots()
    leg = ['loss',]
    ax.plot(x, y, linewidth=2)
    ax.scatter(epoch_end_x, epoch_end_y)
    title = 'Training loss'
    if vdf is not None:
        if len(vdf) > epochs:
            vy = vdf.groupby('batch_id').mean()['validation_loss'].values
            vx = vdf['batch_id'].unique()
        else:
            vy = vdf['validation_loss'].values
            vx = epoch_end_x
        title = title + ' with validation loss'
        leg.append('validation loss')
        if len(vdf) > epochs:
            #vy_err = v_df.groupby('batch_id').sem()['validation_loss'].values
            #ax.errorbar(vx, vy, vy_err, marker='.')
            ax.plot(vx, vy, linewidth=2, marker='o')
        else:
            ax.plot(vx, vy, linewidth=2, marker='o')
    ax.set(xlabel=x_lab, ylabel=y_lab)
    ax.set_title(title)
    ax.legend(leg)
    fig.set_size_inches(13, 9)
    if save is not None:
        plt.savefig(save, dpi=300)
    if show:
        plt.show()
    return fig, ax



def save_channel_loss_plot(path, show=True):
    df = pd.read_csv(path)
    p = Path(path)
    n = p.stem
    d = p.parents[0]
    out_path = os.path.join(d, n + '_channel-loss.png')
    fig, ax = plot_channel_losses(df, save=out_path, show=show)



def plot_channel_losses(df, x_lab='Iteration', y_lab='BCE Loss', save=None, show=True):
    cols = list(df.columns)
    x = df['Unnamed: 0'].values
    non_channel_cols = ['Unnamed: 0', 'epoch', 'batch_num', 'loss', 'data_id']
    channel_losses = [col for col in cols if col not in non_channel_cols]
    fig, axs = plt.subplots(2, 2)
    zs, ys, xs, cs = [], [], [], []
    for col in channel_losses:
        y = df[col].values
        if col.startswith('z'):
            ls = _get_linestyle(zs)
            axs[0, 0].plot(x, y, linewidth=1, linestyle=ls)
            zs.append(col)
        if col.startswith('y'):
            ls = _get_linestyle(ys)
            axs[0, 1].plot(x, y, linewidth=1, linestyle=ls)
            ys.append(col)
        if col.startswith('x'):
            ls = _get_linestyle(xs)
            axs[1, 0].plot(x, y, linewidth=1, linestyle=ls)
            xs.append(col)
        if col.startswith('centre'):
            ls = _get_linestyle(cs)
            axs[1, 1].plot(x, y, linewidth=1, linestyle=ls)
            cs.append(col)
    axs[0, 0].set_title('Z affinities losses')
    axs[0, 0].legend(zs)
    axs[0, 1].set_title('Y affinities losses')
    axs[0, 1].legend(ys)
    axs[1, 0].set_title('X affinities losses')
    axs[1, 0].legend(xs)
    axs[1, 1].set_title('Centreness losses')
    axs[1, 1].legend(cs)
    for ax in axs.flat:
        ax.set(xlabel=x_lab, ylabel=y_lab)
    fig.set_size_inches(13, 9)
    if save is not None:
        plt.savefig(save, dpi=300)
    if show:
        plt.show()
    return fig, axs


def _get_linestyle(lis):
    if len(lis) == 0:
        ls = '-'
    elif len(lis) == 1:
        ls = '--'
    else:
        ls = ':'
    return ls


# --------
# VI Plots
# --------


def VI_plot(
            path, 
            cond_ent_over="GT | Output",  
            cond_ent_under="Output | GT", 
            lab="",
            save=False, 
            show=True):
    df = pd.read_csv(path)
    overseg = df[cond_ent_over].values
    o_groups = [cond_ent_over] * len(overseg)
    underseg = df[cond_ent_under].values
    u_groups = [cond_ent_under] * len(underseg)
    groups = o_groups + u_groups
    x = 'Variation of information'
    y = 'Conditional entropy'
    data = {
        x : groups, 
        y : np.concatenate([overseg, underseg])
        }
    data = pd.DataFrame(data)
    o = 'h'
    pal = 'Set2'
    sigma = .2
    f, ax = plt.subplots(figsize=(12, 10))
    pt.RainCloud(x = x, y = y, data = data, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = o)
    p = Path(path)
    plt.title(p.stem)
    if save:
        save_path = os.path.join(p.parents[0], p.stem + lab + '_VI_rainclout_plot.png')
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def experiment_VI_plots(
        paths, 
        names, 
        title,
        out_name,
        out_dir,
        cond_ent_over="GT | Output",  
        cond_ent_under="Output | GT", 
    ):
    groups = []
    ce0 = []
    ce1 = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        ce0.append(df[cond_ent_over].values)
        ce1.append(df[cond_ent_under].values)
        groups += [names[i]] * len(df)
    x = 'Experiment'
    data = {
        x : groups, 
        cond_ent_over : np.concatenate(ce0), 
        cond_ent_under : np.concatenate(ce1)
    }
    data = pd.DataFrame(data)
    f, axs = plt.subplots(1, 2, figsize=(12, 10))
    ax0 = axs[0, 0]
    ax1 = axs[0, 1]
    o = 'h'
    pal = 'Set2'
    sigma = .2
    pt.RainCloud(x = x, y = cond_ent_over, data = data, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax0, orient = o)
    pt.RainCloud(x = x, y = cond_ent_under, data = data, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax1, orient = o)
    plt.title(title)
    if save:
        save_path = os.path.join(out_dir, '_VI_rainclould_plots.png')
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()



if __name__ == '__main__':
    #name = 'loss_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl.csv'
    name = 'loss_210401_150158_z-1_y-1_x-1__wBCE2-1-1.csv'
    #dir_ = '/Users/amcg0011/Data/pia-tracking/cang_training/210331_training_0'
    dir_ = '/Users/amcg0011/Data/pia-tracking/cang_training/210401_150158_z-1_y-1_x-1__wBCE2-1-1'
    path = os.path.join(dir_, name)
    save_channel_loss_plot(path)
    #v_name = 'validation-loss_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl.csv'
    v_name = 'validation-loss_210401_150158_z-1_y-1_x-1__wBCE2-1-1.csv'
    v_path = os.path.join(dir_, v_name)
    loss_function = 'Weighted BCE Loss (2, 1, 1)'
    save_loss_plot(path, loss_function, v_path)
