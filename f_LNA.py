# -*- encoding: utf-8 -*-
"""
This file contains the main functions used in ALMA-ECOGAL experiment testing LNAs @ ESO.

@author: Francesco Andreetto

May 21st 2024, Garching Bei München (Germany) - May 27th 2024, Brescia (Italy)
"""

# Libraries & Modules
import os
import logging
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def Load_LNAs_Data(path: str, ESO_dataset: bool) -> []:
    """
        Load the data from the all the txt files in a given path\n
        Parameters:\n
        - **path** (``str``): path of the directory containing the txt files with the dataset;
        - **ESO_dataset** (``bool``): specify the kind of LNAs we can find in the dataset (ESO or LNF)\n
        Return\n
        Two lists:
         names: containing the names of the dataset of the txt files
         data: containing the dataset of the txt files
    """
    # Get a list of all files in the directory
    all_files = os.listdir(path)

    # Create a list to include all .txt files
    file_names = [file for file in all_files if file.endswith('.txt')]

    # Create a list to store names without the .txt extension
    names = [os.path.splitext(file)[0] for file in file_names]

    # Initialize an empty list to contain dataset
    data = []
    # Check the kind of dataset to load it properly
    skiprows = 1 if ESO_dataset else 9

    # logging.info(f"The names of the files will be now printed to video:\n{file_names}")

    for filename in file_names:
        # Load data from txt files, skipping "1" row for ESO data, "9" rows for LNF data
        d = np.loadtxt(f"{path}/{filename}", skiprows=skiprows)
        # Collect loaded data
        data.append(d)

    return names, data


def Plot_Single_Noise_Gain(data: list, names: list, ESO_dataset: bool, noise_spec: int,
                           output_plot_dir: str, show: bool):
    """
    Plot the Noise Temperature and the Gain of a set of LNAs and save a png picture of it.\n
    Parameters:\n
    - **data** (``list``): contains the dataset with Noise Temperatures and Gains;\n
    - **names** (``list``): contains the names of the LNAs plotted;\n
    - **ESO_dataset** (``bool``): specify the kind of LNAs we can find in the dataset (ESO or LNF);\n
    - **noise_spec** (``int``): noise specifics (at 80% or 100%) will be printed on the plot;\n
    - **output_plot_dir** (`str`): path from the pipeline dir to the dir that contains the plots of the analysis;\n
    - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only.
    """

    for i in range(len(data)):

        # Set plot name
        plot_name = f"Single_Plot_{names[i]}"
        # Preparing figure to plot
        fig, ax1 = plt.subplots(figsize=(20, 20))

        # Set plot title
        plot_title = f"Noise Temperature and Gain of LNAs\n{names[i]}\n\n"

        # Checking the dataset to plot properly the dataset:
        # Noise Temperature column: "3" for ESO data, "1" for LNF
        # Gain column: "1" for ESO, "2" for LNF
        NT_column, G_column = (3, 1) if ESO_dataset else (1, 2)

        # ==============================================================================================================
        # Noise Temperature
        # ==============================================================================================================

        # Plotting Frequency (column 0) vs Noise Temperature (NT_column)
        logging.debug(names[i])
        ax1.plot(data[i][:, 0], data[i][:, NT_column], linewidth=2, color="royalblue",
                 label="Noise Temperature (corrected)")

        # ==============================================================================================================
        # Gain
        # ==============================================================================================================

        # Plotting Frequency (column 0) vs Gain (G_column)
        logging.debug(names[i])
        ax1.plot(data[i][:, 0], data[i][:, G_column], linewidth=2, color="firebrick",
                 label="Gain")

        # =============================================================================
        # Noise Specifics
        if noise_spec == 80:
            ax1.plot([67, 90], [22, 22], color='k', linestyle='--', linewidth=2)
            ax1.text(80, 23, "Noise Specs, 80%", horizontalalignment='center', fontsize='25')
            ax1.plot([90, 116], [29, 29], color='k', linestyle='--', linewidth=2)
            ax1.text(105, 30, "Noise Specs, 80%", horizontalalignment='center', fontsize='25')
            plot_name += f"_NS_{noise_spec}"

        elif noise_spec == 100:
            ax1.plot([67, 90], [26, 26], color='k', linestyle='-', linewidth=2)
            ax1.text(80, 27, "Noise Specs, 100%", horizontalalignment='center', fontsize='25')
            ax1.plot([90, 116], [33, 33], color='k', linestyle='-', linewidth=2)
            ax1.text(105, 34, "Noise Specs, 100%", horizontalalignment='center', fontsize='25')
            plot_name += f"_NS_{noise_spec}"

        else:
            pass
        # =============================================================================

        # Make legend for the plot
        plt.subplots_adjust(bottom=0.2)
        plt.legend(loc='upper center', shadow=True, fontsize=30, bbox_to_anchor=(0.5, -0.1))
        plt.tick_params(axis='both', which='major', labelsize=20, color="red")

        # Set plot title
        plot_title = f"ESO - {plot_title}" if ESO_dataset else f"LNF - {plot_title}"
        ax1.set_title(f'{plot_title}', fontsize=20)

        # Insert a grid in the plot
        ax1.grid(True)
        # Set axis limits
        ax1.axis([67, 116, 0, 60])
        # Set axis labels
        ax1.set_xlabel("\nFrequency [GHz]\n", fontsize='25', color="black")
        ax1.set_ylabel("\nNoise temperature [K]\n", fontsize='25', color="royalblue")

        # Create a twin axis on the right side for the Gain
        ax2 = ax1.twinx()
        # Set axis limits
        ax2.axis([67, 116, 0, 60])  # necessary?
        # Set axis label
        ax2.set_ylabel("\nGain [dB]\n", fontsize='25', color="firebrick")

        # Checking existence of the dir
        Path(output_plot_dir).mkdir(parents=True, exist_ok=True)
        # Save Figure of the plot
        plt.savefig(f'{output_plot_dir}/{plot_name}.png')
        # Close the Figure to save memory
        plt.close()

        # Show the plot on video if asked
        if show:
            plt.show()


def Plot_All_Noise_Gain(data: list, names: list, ESO_dataset: bool, noise_spec: int,
                        output_plot_dir: str, plot_name: str, plot_title: str, show: bool):
    """
    Plot the Noise Temperature and the Gain of a set of LNAs and save a png picture of it.\n
    Parameters:\n
    - **data** (``list``): contains the dataset with Noise Temperatures and Gains;\n
    - **names** (``list``): contains the names of the LNAs plotted;\n
    - **ESO_dataset** (``bool``): specify the kind of LNAs we can find in the dataset (ESO or LNF);\n
    - **noise_spec** (``int``): noise specifics (at 80% or 100%) will be printed on the plot;\n
    - **output_plot_dir** (`str`): path from the pipeline dir to the dir that contains the plots of the analysis;\n
    - **plot_name** (`str`): name of the plot;\n
    - **plot_title** (`str`): title of the plot;\n
    - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only.
    """
    # Preparing figure to plot
    fig, ax1 = plt.subplots(figsize=(20, 20))

    # Checking the dataset to plot properly the dataset:
    # Noise Temperature column: "3" for ESO data, "1" for LNF
    # Gain column: "1" for ESO, "2" for LNF
    NT_column, G_column = (3, 1) if ESO_dataset else (1, 2)

    # ==================================================================================================================
    # Noise Temperature
    # ==================================================================================================================
    for i in range(0, len(data)):
        # Plotting Frequency (column 0) vs Noise Temperature (NT_column)
        logging.debug(names[i])
        plt.plot(data[i][:, 0], data[i][:, NT_column], linewidth=2, label=names[i])

    # ==================================================================================================================
    # Gain
    # ==================================================================================================================
    for i in range(0, len(data)):
        # Plotting Frequency (column 0) vs Gain (G_column)
        logging.debug(names[i])
        plt.plot(data[i][:, 0], data[i][:, G_column], linewidth=2, label=names[i])

    # Make legend for the plot
    plt.subplots_adjust(bottom=0.4)
    plt.legend(loc='upper center', shadow=True, fontsize='17', bbox_to_anchor=(0.5, -0.1))
    plt.tick_params(axis='both', which='major', labelsize=20, color="red")

    # =============================================================================
    # Noise Specifics
    if noise_spec == 80:
        ax1.plot([67, 90], [22, 22], color='k', linestyle='--', linewidth=2)
        ax1.text(80, 23, "Noise Specs, 80%", horizontalalignment='center', fontsize='25')
        ax1.plot([90, 116], [29, 29], color='k', linestyle='--', linewidth=2)
        ax1.text(105, 30, "Noise Specs, 80%", horizontalalignment='center', fontsize='25')
        plot_name += f"{noise_spec}"

    elif noise_spec == 100:
        ax1.plot([67, 90], [26, 26], color='k', linestyle='-', linewidth=2)
        ax1.text(80, 27, "Noise Specs, 100%", horizontalalignment='center', fontsize='25')
        ax1.plot([90, 116], [33, 33], color='k', linestyle='-', linewidth=2)
        ax1.text(105, 34, "Noise Specs, 100%", horizontalalignment='center', fontsize='25')
        plot_name += f"{noise_spec}"

    else:
        pass
    # =============================================================================

    # Set plot title
    plot_title = f"ESO - {plot_title}" if ESO_dataset else f"LNF - {plot_title}"
    ax1.set_title(f'{plot_title}', fontsize=30)

    # Insert a grid in the plot
    ax1.grid(True)
    # Set axis limits
    ax1.axis([67, 116, 0, 60])
    # Set axis labels
    ax1.set_xlabel("Frequency [GHz]", fontsize='25', color="black")
    ax1.set_ylabel("Noise temperature [K]", fontsize='25', color="black")

    # Create a twin axis on the right side for the Gain
    ax2 = ax1.twinx()
    # Set axis limits
    ax2.axis([67, 116, 0, 60])  # necessary?
    # Set axis label
    ax2.set_ylabel("Gain [dB]", fontsize='25', color="black")

    # Checking existence of the dir
    Path(output_plot_dir).mkdir(parents=True, exist_ok=True)
    # Save Figure of the plot
    plt.savefig(f'{output_plot_dir}/{plot_name}.png')
    # Close the Figure to save memory
    plt.close()

    # Show the plot on video if asked
    if show:
        plt.show()
