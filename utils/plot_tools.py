import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from scipy.interpolate import griddata

# 全局设置，应用于所有图形
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 设置全局字体
    'font.size': 18,                   # 设置全局字体大小
    'axes.titlesize': 22,              # 设置标题字体大小
    'axes.labelsize': 30,              # 设置坐标轴标签字体大小
    'axes.linewidth': 1.5,               # 加粗坐标轴线条
    'lines.linewidth': 3.0,              # 加粗绘制的线条
    'legend.fontsize': 20,             # 设置图例字体大小
    'legend.frameon': False,           # 去除图例边框
    'xtick.labelsize': 20,             # 设置x轴刻度字体大小
    'ytick.labelsize': 20,             # 设置y轴刻度字体大小
    'xtick.major.width': 1.5,            # 设置x轴刻度线宽度
    'ytick.major.width': 1.5,            # 设置y轴刻度线宽度
    'figure.dpi': 80,                  # 设置图形分辨率
    'figure.figsize': [8, 6],          # 设置图形大小
})


def plot1d(
    x,
    y,
    x_test,
    y_test,
    y_mean,
    y_std,
    xlim=None,
    ylim=None,
    xlabel="$x$",
    ylabel="$y$",
    title="",
    save_path=None,
    dpi=300
):
    plt.figure()
    plt.plot(x_test, y_test, "k-", label="Truth")
    plt.plot(x_test, y_mean, "b-", label="Mean")
    plt.fill_between(
        x_test.ravel(),
        y_mean.ravel() + 2 * y_std.ravel(),
        y_mean.ravel() - 2 * y_std.ravel(),
        alpha=0.3,
        facecolor="c",
        label="2 Stds",
    )
    plt.plot(x, y, "r*", markersize=15, label="Training Data")
    plt.legend(ncol=2,
               loc="upper left",
               fontsize=20,  # 字体大小
               handlelength=1,  # 图例中线段长度
               borderpad=0.3,  # 图例框内边距
               labelspacing=0.2,  # 标签之间的垂直间距
               columnspacing=0.5  # 列之间的水平间距
               )
    plt.xlabel(xlabel, fontstyle='italic')
    plt.ylabel(ylabel, fontstyle='italic')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.tight_layout()
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_predictions(
    x_test,
    y_test,
    y_pred,
    xlim=None,
    ylim=None,
    xlabel="$x$",
    ylabel="$y$",
    title="",
    save_path=None,
    dpi=300
):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(x_test, y_test, "k-", label="Exact")  # 绘制真实值
    plt.plot(x_test, y_pred, "r--", label="Predicted")  # 绘制预测值
    plt.legend(
        ncol=1,
        loc="upper left",
        fontsize=20,  # 字体大小
        handlelength=1.3,  # 图例中线段长度
        borderpad=0.3,  # 图例框内边距
        labelspacing=0.2,  # 标签之间的垂直间距
        columnspacing=0.5  # 列之间的水平间距
    )
    plt.xlabel(xlabel, fontstyle='italic')
    plt.ylabel(ylabel, fontstyle='italic')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()


def plot2d(xx1, xx2, y, train_x1=None, train_x2=None, marker='o', xlim=None, ylim=None, xticks_num=None, yticks_num=None,
           clim=None, bar_ticks=None, title="", save_path=None, dpi=300):
    """
    2-D plot.

        Args:
            x1 (array): The 2-D array representing the grid of the first coordinate,
                    with shape [N1, N2].
            x2 (array): The 2-D array representing the grid of the second coordinate,
                    with shape [N1, N2].
            y (array): The 2-D array representing values of the output on the
                    grid formed by x1, x2, with shape [N1, N2].
            train_x1 (array): The 1-D array representing the x-coordinates of the training points.
            train_x2 (array): The 1-D array representing the y-coordinates of the training points.
    """
    fig, ax = plt.subplots(dpi=100)
    c = ax.pcolormesh(xx1, xx2, y, cmap="jet")

    # Optionally overlay training points if provided
    if train_x1 is not None and train_x2 is not None:
        ax.scatter(train_x1, train_x2, color='k', marker=marker, s=60, label='Training Data', edgecolor=None)

    # ax.set_xlabel("$x_1$")
    # ax.set_ylabel("$x_2$")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set number of ticks for x and y axes
    if xticks_num is not None:
        # Generate tick positions including the boundaries
        x_ticks = np.linspace(xlim[0], xlim[1], xticks_num)
        ax.set_xticks(x_ticks)
    if yticks_num is not None:
        # Generate tick positions including the boundaries
        y_ticks = np.linspace(ylim[0], ylim[1], yticks_num)
        ax.set_yticks(y_ticks)

    ax.set_title(title)
    colorbar = fig.colorbar(c, ax=ax)
    if bar_ticks is not None:
        colorbar.set_ticks(bar_ticks)
    c.set_clim(clim)
    plt.tight_layout()
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot2d_points(x, y, value, train_x=None, train_y=None, marker='o', clim=None, title="u"):
    """
     Plot the images by interpolating the points and overlay training points.
    :param x: location x shape=(N, )
    :param y: location y  shape=(N, )
    :param value: value of function  shape=(N, )
    :param train_x: training point x coordinates shape=(M, ) (optional)
    :param train_y: training point y coordinates shape=(M, ) (optional)
    :param marker: marker style for training points (default is 'o')
    :param clim: color limits for the colormap, specified as a tuple (vmin, vmax).
                 If None, automatic scaling is used. Example: (-1, 1)
    :return: None
    """

    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_value = griddata((x, y), value, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(8, 6), dpi=100)
    mesh = plt.pcolormesh(grid_x, grid_y, grid_value, cmap='jet')
    plt.colorbar(mesh)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    # Overlay training points if provided
    if train_x is not None and train_y is not None:
        plt.scatter(train_x, train_y, color='k', marker=marker, s=60, label='Training Data')
        # plt.legend()

    mesh.set_clim(clim)
    plt.tight_layout()
    plt.show()


def plot_loss_old(loss_values, xlabel="Iteration", ylabel="Loss", title="Training Loss"):
    """
    Plot the loss values over iterations.

    :param loss_values: List or array-like of loss values.
    :param xlabel: Label for the x-axis (default is "Iteration").
    :param ylabel: Label for the y-axis (default is "Loss").
    :param title: Title of the plot (default is "Training Loss").
    :return: None
    """

    # Ensure that loss_values is a list or array-like
    if not hasattr(loss_values, '__iter__'):
        raise ValueError("loss_values must be a list or array-like")

    num = len(loss_values)

    fig, ax = plt.subplots(dpi=100)
    ax.plot(range(1, num + 1), loss_values, 'b-')  # Add markers for better visibility  marker='o'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Optional: Add a grid for better readability
    ax.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_loss(loss_values, xlabel="Iteration", ylabel="Loss", title="Training Loss", xlim=None, ylim=None, xticks=None, save_path=None, dpi=300):
    """
    Plot the loss values over iterations with options for custom axis range and scientific notation for the y-axis.

    :param loss_values: List or array-like of loss values.
    :param xlabel: Label for the x-axis (default is "Iteration").
    :param ylabel: Label for the y-axis (default is "Loss").
    :param title: Title of the plot (default is "Training Loss").
    :param xlim: Tuple for x-axis limits (default is None).
    :param ylim: Tuple for y-axis limits (default is None).
    :param xticks: List of x-axis tick positions to display (default is None, which means automatic ticks).
    :param save_path: Path to save the plot image (default is None, which means no saving).
    :param dpi: Resolution of the saved image (default is 300).
    :return: None
    """

    # Ensure that loss_values is a list or array-like
    if not hasattr(loss_values, '__iter__'):
        raise ValueError("loss_values must be a list or array-like")

    num = len(loss_values)

    fig, ax = plt.subplots(dpi=100)
    ax.plot(range(1, num + 1), loss_values, 'b-')  # Add markers for better visibility marker='o'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Set custom x-axis tick positions if provided
    if xticks is not None:
        ax.set_xticks(xticks)  # Set specified tick positions

    # Enable scientific notation for y-axis
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Add a grid with dashed lines
    ax.grid(True, linestyle='--')

    # Show the plot
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()