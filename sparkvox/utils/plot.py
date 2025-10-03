import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_hist_data(
    records,
    title,
    xlabel,
    save_path,
    ylabel="Frequency",
    bins=20,
    color="blue",
    alpha=0.7,
    fontsize=16,
    style=None
):
    """
    Function to extract data and plot histograms.

    Args:
        records (list): Data to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        save_path (str): Path to save the plot image.
        ylabel (str): Label for the y-axis. Default is 'Frequency'.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.set(style="whitegrid")

    # Plot histogram
    ax.hist(records, bins=bins, color=color, alpha=alpha)

    # Set plot title and labels with Arial font
    ax.set_title(title, fontname='Arial', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel(xlabel, fontname='Arial', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontname='Arial', fontsize=fontsize)
    if style=='sci':
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
    
    # Change the font of the tick labels to Arial
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    plt.xticks(fontname='Arial')
    plt.yticks(fontname='Arial')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory


def plot_bar_figure(
    x,
    y,
    title,
    xlabel,
    save_path,
    ylabel="Frequency",
    color="blue",
    alpha=0.7,
    fontsize=16,
    style=None
):
    """
    Function to extract data and plot bar charts.

    Args:
        x: value
        y: label
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        save_path (str): Path to save the plot image.
        ylabel (str): Label for the y-axis. Default is 'Frequency'.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.set(style="whitegrid")

    # Plot bar chart
    ax.bar(x, y, color=color, alpha=alpha)

    # Set plot title and labels with Arial font
    ax.set_title(title, fontname='Arial', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel(xlabel, fontname='Arial', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontname='Arial', fontsize=fontsize)
    if style=='sci':
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
    
    # Change the font of the tick labels to Arial
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    plt.xticks(fontname='Arial', rotation='vertical')  # Rotate x-ticks for better readability
    plt.yticks(fontname='Arial')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory


def plot_line_chart(
    x,
    y,
    title,
    xlabel,
    save_path,
    ylabel="Frequency",
    color="blue",
    alpha=0.7,
    fontsize=16,
    style=None,
    marker='o',
    linestyle='-'
):
    """
    Function to extract data and plot line charts.

    Args:
        records (list): Data to plot, should be a list of tuples or lists where first element is x and second is y.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        save_path (str): Path to save the plot image.
        ylabel (str): Label for the y-axis. Default is 'Frequency'.
        marker (str): Marker style for the line plot. Default is 'o'.
        linestyle (str): Line style for the line plot. Default is '-'.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.set(style="whitegrid")

    # Plot line chart
    ax.plot(x, y, marker=marker, linestyle=linestyle, color=color, alpha=alpha)

    # Set plot title and labels with Arial font
    ax.set_title(title, fontname='Arial', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel(xlabel, fontname='Arial', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontname='Arial', fontsize=fontsize)
    if style == 'sci':
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    # Set tick parameters
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)

    # Change the font of the tick labels to Arial
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    plt.xticks(fontname='Arial')
    plt.yticks(fontname='Arial')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory



def plot_density_data(
    records_list,
    title,
    xlabel,
    save_path,
    ylabel="Density",
    color_list=None,
    alpha=0.7,
    fontsize=16,
    style='sci',
    legend_labels=None
):
    """
    Function to extract data and plot density (KDE) curves with shaded area under the curve.

    Args:
        records_list (list of lists): A list containing multiple datasets to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        save_path (str): Path to save the plot image.
        ylabel (str): Label for the y-axis. Default is 'Density'.
        color_list (list of str): List of colors to use for the different datasets. Default is None, which uses a default color palette.
        alpha (float): Transparency of the shaded area. Default is 0.7.
        fontsize (int): Font size for the labels and title. Default is 16.
        style (str): If 'sci', use scientific notation for y-axis. Default is None.
        legend_labels (list of str): Labels for the legend. Default is None, which uses indices.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.set(style="whitegrid")

    # If color_list is not provided, use seaborn's color palette
    if color_list is None:
        color_list = sns.color_palette("Set1", len(records_list))

    # Plot density for each dataset
    for i, records in enumerate(records_list):
        sns.kdeplot(
            records,
            color=color_list[i], 
            shade=True,  # Fill the area under the curve
            alpha=alpha,  # Transparency of the shaded area
            label=legend_labels[i] if legend_labels else f'Dataset {i+1}',
            cut=0
        )
    
    # Set plot title and labels with Arial font
    ax.set_title(title, fontname='Arial', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel(xlabel, fontname='Arial', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontname='Arial', fontsize=fontsize)

    if style == 'sci':
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    # Set tick parameters
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
    
    # Change the font of the tick labels to Arial
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    plt.xticks(fontname='Arial')
    plt.yticks(fontname='Arial')

    # Add legend
    ax.legend(fontsize=fontsize, loc='upper right')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory


def plot_cdf_data(
    records_list,
    title,
    xlabel,
    save_path,
    ylabel="CDF",
    color_list=None,
    alpha=0.7,
    fontsize=16,
    style='sci',
    legend_labels=None
):
    """
    Function to extract data and plot Cumulative Distribution Function (CDF) curves with shaded area under the curve.

    Args:
        records_list (list of lists): A list containing multiple datasets to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        save_path (str): Path to save the plot image.
        ylabel (str): Label for the y-axis. Default is 'CDF'.
        color_list (list of str): List of colors to use for the different datasets. Default is None, which uses a default color palette.
        alpha (float): Transparency of the shaded area. Default is 0.7.
        fontsize (int): Font size for the labels and title. Default is 16.
        style (str): If 'sci', use scientific notation for y-axis. Default is None.
        legend_labels (list of str): Labels for the legend. Default is None, which uses indices.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.set(style="whitegrid")

    # If color_list is not provided, use seaborn's color palette
    if color_list is None:
        color_list = sns.color_palette("Set1", len(records_list))

    # Plot CDF for each dataset
    for i, records in enumerate(records_list):
        # Sort data for CDF plot
        sorted_data = np.sort(records)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        # Plot CDF
        ax.plot(sorted_data, cdf, color=color_list[i], lw=2, label=legend_labels[i] if legend_labels else f'Dataset {i+1}')

    # Set plot title and labels with Arial font
    ax.set_title(title, fontname='Arial', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel(xlabel, fontname='Arial', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontname='Arial', fontsize=fontsize)

    if style == 'sci':
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    # Set tick parameters
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
    
    # Change the font of the tick labels to Arial
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    plt.xticks(fontname='Arial')
    plt.yticks(fontname='Arial')

    # Add legend
    ax.legend(fontsize=fontsize, loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
