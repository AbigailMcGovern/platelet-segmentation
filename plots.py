import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path


def save_loss_plot(path, loss_function, v_path=None):
    df = pd.read_csv(path)
    p = Path(path)
    n = p.stem
    d = p.parents[0]
    out_path = os.path.join(d, n + '_loss.png')
    fig, ax = plot_loss(df, vdf=None, x_lab='Iteration', y_lab=loss_function, save=out_path)



def plot_loss(df, vdf=None, x_lab='Iteration', y_lab='BCE Loss', save=None):
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
    if v_path is not None:
        v_df = pd.read_csv(v_path)
        vx = epoch_end_x
        vy = v_df['validation_loss'].values
        title = title + ' with validation loss'
        leg.append('validation loss')
        ax.plot(vx, vy, linewidth=2, marker='o')
    ax.set(xlabel=x_lab, ylabel=y_lab)
    ax.set_title(title)
    ax.legend(leg)
    fig.set_size_inches(13, 9)
    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()
    return fig, ax



def save_channel_loss_plot(path):
    df = pd.read_csv(path)
    p = Path(path)
    n = p.stem
    d = p.parents[0]
    out_path = os.path.join(d, n + '_channel-loss.png')
    fig, ax = plot_channel_losses(df, save=out_path)



def plot_channel_losses(df, x_lab='Iteration', y_lab='BCE Loss', save=None):
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


if __name__ == '__main__':
    name = 'loss_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c.csv'
    dir_ = '/Users/amcg0011/Data/pia-tracking/cang_training/210323_training_0'
    path = os.path.join(dir_, name)
    save_channel_loss_plot(path)
    v_name = 'validation-loss_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c.csv'
    v_path = os.path.join(dir_, v_name)
    loss_function = 'BCE Loss'
    save_loss_plot(path, loss_function, v_path)
