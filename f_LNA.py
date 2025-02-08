# -*- encoding: utf-8 -*-
"""
This file contains the main functions used in ALMA-ECOGAL experiment testing LNAs @ ESO.
Classes for LNAs and Chained LNAs are also present to better handle the data collected in the lab.
@author: Francesco Andreetto

May 21st 2024, Garching Bei München (Germany) - February 8th 2025, Garching Bei München (Germany)
"""

# Libraries & Modules
import os
import logging
import math
import random

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from scipy.stats import linregress


########################################################################################################################
#                                   CLASS LNA: MPI
########################################################################################################################
class MPI:

    def __init__(self, name: str):
        """
            Constructor. Class for an MPI amplifier.
        """

        # Name of the LNA
        self.name = name
        # Gain data
        # -------------------------
        # Frequencies
        self.fg = []
        # Gain
        self.gain = []
        # -------------------------

        # Noise T data
        # -------------------------
        # Frequencies
        self.fn = []
        # Noise Temperature
        self.noise_T = []
        # -------------------------

    def load_LNA(self, path_data_file: str):
        """
            Parameters:\n
                 - **path_data_file** (``str``): location of the data file where gain (S21) and noise values are stored
        """
        # Get Gain file
        file = Get_File(path_data_file=path_data_file, keyword1=self.name, keyword2="sParam")
        logging.debug(f"Found MPI file in the folder:\n{file}\n")
        # Open Gain file and read all the lines
        logging.debug("Loading Gain values.\n")
        self.fg, self.gain, _ = Load_Columns(path_data_file=f"{path_data_file}{file}",
                                             col1=0, col2=2, skip_lines=21)

        # Get Noise file
        file = Get_File(path_data_file=path_data_file, keyword1=self.name, keyword2="noise")
        # Open Noise file and read all the lines
        logging.debug("Loading Noise Temperature values.\n")
        self.fn, self.noise_T, _ = Load_Columns(path_data_file=f"{path_data_file}{file}",
                                                col1=0, col2=1, skip_lines=21)

    def clean_LNA(self):
        """
        Remove the frequencies and the corresponding values out of the Band 2: 67GHz - 116GHz
        """
        self.gain = [g for g, fg in zip(self.gain, self.fg) if fg >= 67.0]
        self.noise_T = [n for n, fn in zip(self.noise_T, self.fn) if fn >= 67.0]
        self.fg = [fg for fg in self.fg if fg >= 67.0]
        self.fn = [fn for fn in self.fn if fn >= 67.0]

    def prepare_LNA(self):
        """
        Prepare the MPI LNA binning the values of the gain to have equally spaced dataset (every 0.2 GHz)
        """
        # Initialize an empty lists
        new_g = []
        step_g = []
        # Create the new frequency array
        new_f = np.arange(start=67.0, stop=116.1, step=0.2)

        # Fill the new gain list with the mean value in between each frequency range
        for idx, n_f in enumerate(new_f[:-1]):
            step = idx
            for f, g in zip(self.fg, self.gain):

                # Define every frequency range and repeat

                # Frequency in the range
                if n_f <= f <= new_f[idx + 1]:
                    # Collect gain value
                    step_g.append(g)
                    # logging.debug(f"Current val: {n_f} <= {f} <= {new_f[idx + 1]}\nThe gains are {len(step_g)}.\n\n")

                # Frequency out of the range
                elif f > new_f[idx + 1] and step == idx:

                    # If a gain value is not sampled
                    if not step_g:
                        # check if there are 10 values before
                        if len(new_g) >= 11:
                            # Adding the mean of the previous 10 values
                            new_g.append(np.mean(new_g[step - 11: step - 1]))
                            # logging.debug(f"Gain valued not sampled for the frequency: {n_f}. "
                            #      f"Adding: {new_g[step-1]}. length: {len(new_g)}.\n\n")
                        else:
                            # Adding the previous value
                            new_g.append(g)
                        step += 1
                        break

                    else:
                        # Mean the values of the gain and store the new value
                        new_g.append(np.mean(step_g))
                        # logging.debug(f"Value added: {new_g[step]}, length: {len(new_g)} - FREQ: {n_f}.\n")
                        step_g = []
                        step += 1
                        break

        # Set the new datasets
        self.fg = new_f
        # Final correction to the gain to have 246 values
        for i in range(246 - len(new_g)):
            # Append the mean of the last 10 values of the gains
            new_g.append(np.mean(new_g[-11:-1]))
        self.gain = new_g

    def plot_LNA(self, output_plot_dir: str, show=True):
        """
        Plot Gain and Noise Temperature of the LNA of MPI.
        Parameters:\n
            - **output_plot_dir** (`str`): path to the dir that stores the plots of the analysis;\n
            - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only.
        """
        Plot(freq_1=self.fn, noise_T=self.noise_T,
             freq_2=self.fg, gain=self.gain,
             name=self.name, output_plot_dir=output_plot_dir, show=show)


########################################################################################################################
#                                   CLASS LNA: LNF
########################################################################################################################
class LNF:

    def __init__(self, name: str):
        """
            Constructor. Class for a LNF amplifier.
        """
        # Name of the LNA
        self.name = name
        # Frequencies
        self.f = []
        # Gain data
        self.gain = []
        # Noise T data
        self.noise_T = []

    def load_LNA(self, path_data_file: str):
        """
        Parameters:
                - **data_file** (``str``): location of the data file where noise and gain values are stored
        """
        # Get file
        file = Get_File(path_data_file=path_data_file, keyword1=self.name, keyword2="LNF")
        logging.debug(f"Found LNF file in the folder:\n{file}\n")

        logging.debug("Loading Frequency, Noise Temperature and Gain values.\n")
        self.f, self.noise_T, self.gain = Load_Columns(path_data_file=f"{path_data_file}{file}",
                                                       col1=0, col2=1, col3=2, skip_lines=9)
        # Convert frequency to GHz
        self.f = [f / 1e9 for f in self.f]

    def clean_LNA(self):
        """
        Remove the frequencies and the corresponding values out of the Band 2: 67GHz - 116GHz
        """
        self.gain = [g for g, f in zip(self.gain, self.f) if f >= 67.0]
        self.noise_T = [n for n, f in zip(self.noise_T, self.f) if f >= 67.0]
        self.f = [f for f in self.f if f >= 67.0]

    def prepare_LNA(self):
        """
        Prepare the LNF LNA adding "unity-values" to have equally spaced dataset (every 0.2 GHz)
        """
        # Initialize two empty lists
        new_g = []
        new_n = []

        # Fill those lists with the gain and noise values
        for g, n, f in zip(self.gain, self.noise_T, self.f):
            new_g.append(g)
            new_n.append(n)
            # Take the value two times when the frequency reaches the unity
            if f % 1 == 0.:
                new_g.append(g)
                new_n.append(n)

        # Set the new datasets
        self.gain = new_g[:-1]
        self.noise_T = new_n[:-1]
        self.f = np.arange(start=67.0, stop=116.1, step=0.2)

    def plot_LNA(self, output_plot_dir: str, show=True):
        """
        Plot Gain and Noise Temperature of the LNA of LNF.
        Parameters:\n
            - **output_plot_dir** (`str`): path to the dir that stores the plots of the analysis;\n
            - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only.
        """
        Plot(freq_1=self.f, noise_T=self.noise_T,
             freq_2=self.f, gain=self.gain,
             name=f"LNF-{self.name}", output_plot_dir=output_plot_dir, show=show)


########################################################################################################################
#                                   CLASS CHAIN OF LNAs
########################################################################################################################
class LNAs_Chain:

    def __init__(self, name_lna_1: str, name_lna_2: str):
        """
            Constructor. Class for a chain of two LNAs.
                Parameters:
                 - **name_lna_1** (``str``): name of the first LNA
                 - **name_lna_2** (``str``): name of the second LNA

        """
        # Name of the LNA 1 of MPI
        self.name_1 = name_lna_1
        self.MPI = MPI(name=name_lna_1)
        # Name of the LNA 2 of LNF
        self.name_2 = name_lna_2
        self.LNF = LNF(name=name_lna_2)

        # Chain Name
        self.name = f"{self.name_1}_LNF-{self.name_2}"
        # Best bias configuration
        self.best_bias_config = ""

        # Real Values from samples in dataset
        # Frequencies
        self.f = []
        # Gain data
        self.gain = []
        # Noise T data
        self.noise_T = []

        # Computed values
        # Expected Frequencies
        self.exp_f = []
        # Expected Gain values
        self.exp_gain = []
        # Expected Noise T values
        self.exp_noise_T = []

        # Compatibility: to collect infos about the quality of the chain
        self.quality = {}

    def get_quality(self, real=False):
        """
        Calculate the values of quality of the chain.\n
            Parameters:\n
                - **real** (``bool``): *True* -> Parse Computed & Real chain values, *False* -> Only computed values.
        NOTE: the chain must be evaluated, also if real is True the chain must be loaded before!
        """
        # Get the quality infos for the computed values
        if not real:
            self.quality.update({"exp_slope": 0,
                                 "exp_flatness": 0,
                                 "exp_compliance": {}})
        # Get the quality infos also for real values
        else:
            self.quality.update({"slope": 0,
                                 "flatness": 0,
                                 "compliance": {}})

        # Calculate Compatibility parameters
        # --------------------------------------------------------------------------------------------------------------
        if not real:
            # Computed Values
            # Get the slope of the regression line that approximates the gain curve
            self.quality["exp_slope"] = Get_Slope(x=self.exp_f, y=self.exp_gain)
            # Get the flatness of the gain curve: the mean absolute deviation
            self.quality["exp_flatness"] = Get_Flatness(dataset=self.exp_gain)
            # Get Noise compliance
            self.quality["exp_compliance"] = Get_Chain_Compliance(data=self.exp_noise_T)

        # Real Data
        else:
            # Get the slope of the regression line that approximates the gain curve
            self.quality["slope"] = Get_Slope(x=self.f, y=self.gain)
            # Get the flatness of the gain curve: the mean absolute deviation
            self.quality["flatness"] = Get_Flatness(dataset=self.gain)
            # Get Noise compliance
            self.quality["compliance"] = Get_Chain_Compliance(data=self.noise_T)

        return self.quality

    def evaluate_chain(self, best_gain: list, best_noise: list):
        """
        Take the best parameter for gain and noise evaluation and evaluate the chain.\n
            Parameters:\n
            - **best_gain** (``list``): list of the best gain parameters found from data of reference chains
            - **best_noise** (``list``): list of the best noise parameters found from data of reference chains
        """
        # Evaluation of the Expected values of the chain

        for g1, g2, g_i, n1, n2, n_i in zip(self.MPI.gain, self.LNF.gain, best_gain,
                                            self.MPI.noise_T, self.LNF.noise_T, best_noise):
            # Gain evaluation
            self.exp_gain.append(g_i * (g1 + g2))
            # Noise T evaluation
            self.exp_noise_T.append(n_i * (n1 + n2))
            # Frequency evaluation
            self.exp_f = np.arange(67., 116.1, 0.2)

    def load_chain(self, path_data_file: str):
        """
        Load the chain with frequency, gain and noise Temperature taking data from the path of the dir given.
            Parameters:\n
                - **path_data_file** (``str``): location of the chain datafile where noise and gain values are stored.
        """
        # Get file
        file = Get_File(path_data_file=path_data_file, keyword1=self.name_1, keyword2=self.name_2)
        logging.info(f"# ---------------------------------------------------------------------------------------------"
                     f"\nFound datafile for a chain in the folder:\n{file}\n"
                     f"# ---------------------------------------------------------------------------------------------")

        self.f, self.gain, self.noise_T = Load_Columns(path_data_file=f"{path_data_file}{file}",
                                                       col1=0, col2=1, col3=3, skip_lines=1)
        self.best_bias_config = file[:66]

    def plot_chain(self, output_plot_dir: str, show=True):
        """
        Plot Gain and Noise Temperature of the cascaded LNAs.
        Parameters:\n
            - **output_plot_dir** (`str`): path to the dir that stores the plots of the analysis;\n
            - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only.
        """
        Plot(freq_1=self.f, noise_T=self.noise_T,
             freq_2=self.f, gain=self.gain,
             name=self.name, output_plot_dir=output_plot_dir, show=show)

    def plot_exp_chain(self, output_plot_dir: str, show: bool, real=False):
        """
        Plot Gain and Noise Temperature of the evaluated chain and, if requested, also of the real values.
        Parameters:\n
            - **output_plot_dir** (`str`): path to the dir that stores the plots of the analysis;\n
            - **real** (``bool``): *True* -> Plot computed & Real chain values, *False* -> Plot only computed values.
            - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only.
        NOTE: for the real datas, tha chain must be loaded before (see method load_chain)
        """
        # Create the figure to plot the expected chain
        fig, ax1 = plt.subplots(figsize=(13, 13))

        # --------------------------------------------------------------------------------------------------------------
        # Expected (computed) values
        # --------------------------------------------------------------------------------------------------------------
        # Expected Regression Line
        # exp_slope = Get_Slope(x=self.exp_f, y=self.exp_gain)
        exp_intercept = Get_Intercept(x=self.exp_f, y=self.exp_gain)
        exp_line = []
        for f in self.exp_f:
            exp_line.append(f * self.quality['exp_slope'] + exp_intercept)

        plt.plot(self.exp_f, exp_line, label=f"Expected Regression")
        # Gain
        plt.plot(self.exp_f, self.exp_gain, label="Expected Gain")
        # Noise Temperature
        plt.plot(self.exp_f, self.exp_noise_T, label="Expected Noise T")

        if real:
            # ----------------------------------------------------------------------------------------------------------
            # Real Data
            # ----------------------------------------------------------------------------------------------------------
            # Regression line
            # slope = Get_Slope(x=self.f, y=self.gain)
            intercept = Get_Intercept(x=self.f, y=self.gain)
            regression_line = []
            for f in self.f:
                regression_line.append(f * self.quality['slope'] + intercept)

            plt.plot(self.f, regression_line, label=f"Data Regression")
            # Gain
            plt.plot(self.f, self.gain, label="Gain")
            # Noise Temperature
            plt.plot(self.f, self.noise_T, label="Noise T")
            # ----------------------------------------------------------------------------------------------------------

        # Noise Compliances lines of 80% and 100%
        ax1.plot([67, 90], [26, 26], color='k', linestyle='-', linewidth=1)
        ax1.text(80, 27, "Noise Specs ~ 100%", horizontalalignment='center', fontsize='8')
        ax1.plot([90, 116], [33, 33], color='k', linestyle='-', linewidth=1)
        ax1.text(105, 34, "Noise Specs ~ 100%", horizontalalignment='center', fontsize='8')
        ax1.plot([67, 90], [22, 22], color='k', linestyle='--', linewidth=1)
        ax1.text(80, 23, "Noise Specs ~ 80%", horizontalalignment='center', fontsize='8')
        ax1.plot([90, 116], [29, 29], color='k', linestyle='--', linewidth=1)
        ax1.text(105, 30, "Noise Specs ~ 80%", horizontalalignment='center', fontsize='8')

        # Get Computed Quality information
        exp_info = (f"Expected Quality\n"
                    f"\n1) Slope: {np.round(self.quality['exp_slope'], 4)}\n"
                    f"\n2) Flatness: {np.round(self.quality['exp_flatness'], 4)}\n"
                    f"\n3) Compliances:\n"
                    f"\nspecs @ 80%: {self.quality['exp_compliance'][0]}"
                    f"\nspecs @ 100%: {self.quality['exp_compliance'][1]}"
                    f"\nNoise T (67-90GHz): {self.quality['exp_compliance'][2]} K"
                    f"\nNoise T (90-116GHz): {self.quality['exp_compliance'][3]} K"
                    )
        if not real:

            # Print the box in the center of the plot
            ax1.text(97, 14, exp_info, style='italic',
                     bbox={'facecolor': 'red', 'alpha': 0.4, 'pad': 10})
        else:
            # Print the box on the left of the plot
            ax1.text(92, 14, exp_info, style='italic',
                     bbox={'facecolor': 'red', 'alpha': 0.4, 'pad': 10})
            # Get Data quality information
            info = (f"Measured Quality\n"
                    f"\n1) Slope: {np.round(self.quality['slope'], 4)}\n"
                    f"\n2) Flatness: {np.round(self.quality['flatness'], 4)}\n"
                    f"\n3) Compliances:\n"
                    f"\nspecs @ 80%: {self.quality['compliance'][0]}"
                    f"\nspecs @ 100%: {self.quality['compliance'][1]}"
                    f"\nNoise T (67-90GHz): {self.quality['compliance'][2]} K"
                    f"\nNoise T (90-116GHz): {self.quality['compliance'][3]} K"
                    )
            # Print the box on the right of the plot
            ax1.text(105, 14, info, style='italic',
                     bbox={'facecolor': 'green', 'alpha': 0.4, 'pad': 10})

        # Plot setup: axis
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.set_ylabel("\nNoise temperature [K]\n", fontsize=15)
        ax1.set_xlabel("\nFrequency [GHz]\n", fontsize=15)
        ax1.set_ylim(12, 55)

        # Create a twin axis on the right side for the Gain
        ax2 = ax1.twinx()
        ax2.set_ylabel("\nGain [dB]\n", fontsize=15)
        ax2.set_ylim(12, 55)
        # Make legend
        ax1.legend()

        # Plot Title & figure Name
        plot_title = f"Chain {self.name}\nExpected values"
        figure_name = f"Evaluation_Chain_{self.name}"

        if real:
            plot_title += f" vs Real Data\n\n{self.best_bias_config}\n"
            figure_name += "_Real"

        plt.title(f"{plot_title}")
        plt.savefig(f"{output_plot_dir}/{figure_name}.png", bbox_inches='tight')

        # Show the plot on video if asked
        if show:
            plt.show()
        else:
            plt.close()

    def prepare_chain(self, path_data_file_MPI: str, path_data_file_LNF: str):
        """
        Load and prepare the chain with the dataset of the MPI and LNF provided.\n
            Parameters:\n
            - **path_data_file_MPI** (``str``): location of the MPI datafile where noise and gain values are stored.
            - **path_data_file_LNF** (``str``): location of the LNF datafile where noise and gain values are stored.
        """
        # --------------------------------------------------------------------------------------------------------------
        # 1. Load and Prepare the chain LNAs
        # Load MPI data
        self.MPI.load_LNA(path_data_file=path_data_file_MPI)
        # Clean MPI LNA
        self.MPI.clean_LNA()
        # Prepare MPI LNA
        self.MPI.prepare_LNA()

        # Load LNF data
        self.LNF.load_LNA(path_data_file=path_data_file_LNF)
        # Clean LNF LNA
        self.LNF.clean_LNA()
        # Prepare LNF LNA
        self.LNF.prepare_LNA()


########################################################################################################################
# FUNCTIONS OF GENERIC USE
########################################################################################################################


def Get_Chain_Compliance(data: list) -> []:
    """
    Return a list containing the following information:\n
    1) Percentage of values at low frequency (67GHz<f<90GHz) below 23.\n
    2) Percentage of values at high frequency (90Ghz<f<116GHz) below 29.\n
    3) Noise Temperature at low frequency (67GHz<f<90GHz);\n
    4) Noise Temperature at high frequency (90Ghz<f<116GHz).\n
    Parameters:\n
        - **output_plot_dir** (`str`): path to the dir that stores the plots of the analysis;\n
        - **show** (``bool``): *True* -> show the plot and save the figure, *False* -> save the figure only.
    """
    compl = [
        # Compliance of 80% of the points - Almost all the points should stay below the specifications
        np.rint((len([x for x in data[:115] if x <= 23.]) + len([x for x in data[115:] if x <= 29.]))
                * 100 / len(data)),
        # Compliance of 100% of the points - All points should stay below the specifications
        np.rint((len([x for x in data[:115] if x <= 27.]) + len([x for x in data[115:] if x <= 34.]))
                * 100 / len(data)),
        # Mean Low Freq
        np.round(np.mean(data[:115]), 1),
        # Mean High Freq
        np.round(np.mean(data[115:]), 1)
    ]

    return compl


def Get_Figure_Of_Merit(slope: float, flatness: float, temperature: float,
                        weight_s=1., weight_f=1., weight_t=1.) -> float:
    """
        Compute a weighted sum between the slope of the linear regression of a curve and the "flatness",
        the discard from the median value of the curve (max-min).\n
        Parameters:\n
        - **slope** (``float``): slope of the line of regression computed on gain data;\n
        - **flatness** (``float``): discard from the median value of the curve (max-min);\n
        - **temperature** (``float``): Noise T at high frequencies;\n
        - **weight_s** (``float``): weight of the slope in the sum;\n
        - **weight_t** (``float``): weight of the flatness in the sum.
    """
    return 10 * slope * weight_s + flatness * weight_f + temperature * weight_t


def Get_File(path_data_file: str, keyword1: str, keyword2: str) -> str:
    """
        Look into a directory if there is a file containing 2 specific keyword.\n
        Parameters:\n
        - **path_data_file** (``str``): path of the directory containing data files;
        - **keyword1** (``str``): first keyword to look for;
        - **keyword2** (``str``): second keyword to look for;
    """
    # Look at all the files in the directory
    for file in os.listdir(path_data_file):
        # files must be txt or dat
        if file.endswith('.txt') or file.endswith('.dat'):
            # Check the keywords
            if keyword1 in file and keyword2 in file:
                return file


def Get_Flatness(dataset: list) -> float:
    """
       Return the difference between the max and the min value of a dataset in respect to the median.\n
       Parameters:\n
       - **dataset** (``list``): list of data.
    """
    # Compute the median value
    m = np.median(dataset)
    # Remove the median as offset to the dataset and normalize
    dataset = [100 * (x - m) / m for x in dataset]
    # Calculate the difference between max and min deviation from the mean
    flatness = np.max(dataset) - np.abs(np.min(dataset))

    return flatness


def Get_Slope(x: list, y: list) -> float:
    """
       Return the slope of the interpolation line of the dataset y=f(x)\n
       Parameters:\n
       - **x**, **y** (``list``): lists of data.
    """
    return linregress(x, y)[0]


def Get_Intercept(x: list, y: list) -> float:
    """
       Return the intercept of the interpolation line of the dataset y=f(x).\n
       Parameters:\n
       - **x**, **y** (``list``): lists of data.
    """
    return linregress(x, y)[1]


def Get_Cascaded_LNAs_names(directory_path: str) -> (list, list):
    """
        Return the names of the LNAs couples of the chains parsing the names of the files in the given directory_path.\n
        Parameters:\n
       - **directory_path** (``str``): path of the directory that contains chains data (gain and noise T).
   """
    # Initialize empty lists for the names
    MPI_names = []
    LNF_names = []

    # Look @ all files in the dir path
    for name_file in os.listdir(directory_path):
        # Look for "Wx." and the 6 following char
        if "Wx." in name_file:
            MPI_index = name_file.find("Wx.")
            # Check if there are at least 6 char after "Wx."
            if MPI_index != -1 and len(name_file) >= MPI_index + 9:
                MPI_names.append(name_file[MPI_index:MPI_index + 9])

        # Look for "LNF-" and the 4 following char
        if "LNF-" in name_file:
            LNF_index = name_file.find("LNF-")
            # Check if there are at least 4 char after "LNF-"
            if LNF_index != -1 and len(name_file) >= LNF_index + 8:
                LNF_names.append(name_file[LNF_index + 4:LNF_index + 8])

    return MPI_names, LNF_names


def Get_LNAs_names(directory_path: str, kind: str) -> list:
    """
        Return the names of the LNAs couples of the chains parsing the names of the files in the given directory_path.\n
        Parameters:\n
       - **directory_path** (``str``): path of the directory that contains chains data (gain and noise T).
       - **kind** (``str``): specify the kind of LNAs (MPI or LNF)
   """
    # Initialize empty lists for the names
    names = []
    if kind == "MPI":
        keyword = "_noise"
        n_char = 9
    else:
        keyword = " Compression"  # Ok for batch 7
        # keyword = " 1 202"  # Ok for batch 8
        n_char = 4
        
    # PUT EXCEPTIONS
    
    # Look @ all files in the dir path
    for name_file in os.listdir(directory_path):

        # Look for the keyword and the previous n_char
        if keyword in name_file:
            index = name_file.find(keyword)
            # Check if there are at least a number of char n_char before the keyword
            if index - n_char != -1:
                names.append(name_file[index - n_char:index])

    return names


def Get_Params(val_1: list, val_2: list, val_tot: list) -> list:
    """
        Return the list of parameters that satisfy the inverse of the following formula:\n
        parameter * (val_1 + val_2) = val_tot
        Parameters:\n
        - **val_1**, **val_2**, **val_tot** (``list``): lists of values (Gains or Noise T).
    """
    # Initialize an empty list of parameters
    params = []
    for v1, v2, vt in zip(val_1, val_2, val_tot):
        params.append(vt / (v1 + v2))
    return params


def Get_Mean_Params(array_array: list):
    """
        Take an array of arrays and compute the mean on every i element of those array.
        Parameters:\n
        - **array_array** (``list``): lists of lists of the same dimension.
    """
    # Mean over columns on the transposed array of array
    mean_val = [np.mean(col) for col in zip(*array_array)]
    return mean_val


def Load_Columns(path_data_file: str,
                 col1: int, col2: int, col3=None, skip_lines=0) -> (list, list, list):
    """
    Load the values of the specified columns of a given file into two or three lists.
    Parameters:\n
    - **path_data_file** (``str``): path of the directory containing data files;
    - **col1** (``int``): column of the file from which extract the values of the first list;
    - **col2** (``int``): column of the file from which extract the values of the second list;
    - **col3** (``int``): column of the file from which extract the values of the third list;
    - **skip_lines** (``int``): number of rows that must be skipped while reading of the file;
    """

    # Initialize empty lists
    list1 = []
    list2 = []
    list3 = []

    # Open Gain file and read all the lines
    with open(f"{path_data_file}", 'r') as file:
        lines = file.readlines()

    # Skip the header of the file
    for line in lines[skip_lines:]:

        # Parse each line
        values = [float(num) for num in line.split()]

        # Store value in the first list
        list1.append(values[col1])
        # Store value in the second list
        list2.append(values[col2])

        if col3:
            # Store value in the third list
            list3.append(values[col3])
    logging.debug("Loading completed.\n")

    return list1, list2, list3


def Plot(freq_1: list, noise_T: list,
         freq_2: list, gain: list,
         name: str,
         output_plot_dir: str,
         show: bool):
    """
    Function that plots stuff in input.
    """
    # Checking existence of the dir
    Path(output_plot_dir).mkdir(parents=True, exist_ok=True)

    # Preparing figure to plot
    fig, ax1 = plt.subplots(figsize=(15, 15))

    # Plotting Frequency vs Noise Temperature
    ax1.plot(freq_1, noise_T, linewidth=2, color="royalblue",
             label="Noise Temperature (corrected)")

    # Plotting Frequency vs Gain
    ax1.plot(freq_2, gain, linewidth=2, color="firebrick",
             label="Gain")

    # Make legend for the plot
    plt.subplots_adjust(bottom=0.2)
    plt.legend(loc='upper center', shadow=True, fontsize=15, bbox_to_anchor=(0.5, -0.15))

    # Set plot title
    plot_title = f"Noise Temperature and Gain of LNA(s): \n{name}\n"
    ax1.set_title(f'{plot_title}', fontsize=20)

    # Insert a grid in the plot
    ax1.grid(True)
    # Set axis limits for Noise T
    ax1.axis([67, 116, 0, 60])
    # Set axis labels
    ax1.set_xlabel("\nFrequency [GHz]\n", fontsize=20, color="black")
    ax1.tick_params(axis='both', which='major', labelsize=15, color="black")
    ax1.set_ylabel("\nNoise temperature [K]\n", fontsize=20, color="royalblue")

    # Create a twin axis on the right side for the Gain
    ax2 = ax1.twinx()
    # Set axis limits for Gain
    ax2.axis([67, 116, 0, 60])
    # Set axis label
    ax2.set_ylabel("\nGain [dB]\n", fontsize=20, color="firebrick")
    ax2.tick_params(axis='both', which='major', labelsize=15, color="black")

    # Checking existence of the dir
    Path(output_plot_dir).mkdir(parents=True, exist_ok=True)
    # Save Figure of the plot
    plot_name = f"{name}_Noise_Gain"
    plt.savefig(f'{output_plot_dir}/{plot_name}.png')
    # Close the Figure to save memory
    plt.close()

    # Show the plot on video if asked
    if show:
        plt.show()


def Write_LNAs_txt_File(vector: list, file_name: str):
    """
    Create a txt file and write the couples of LNAs of kind MPI and LNF in two columns
    Parameters:\n
    - **vector** (``list``): array that contains arrays of two str each containing coupled LNAs names;\n
    - **file_name** (``int``): name and location of the file to open.
    """
    # Open the file in write mode
    with open(file_name, 'w') as file:
        # Write the column headers: not functional for the Report production...
        # file.write("MPI\tLNF\n")

        # Iterate over each pair in the vector
        for pair in vector:
            # Write each pair on the file
            file.write(f"{pair[0]}\n{pair[1]}\n")


########################################################################################################################
# SIMULATED ANNEALING
########################################################################################################################

def calculate_cost(pairs: list, my_dict: dict) -> float:
    """
    Function that calculates the total cost (sum of values) of a combination of LNAs.\n
    Parameters:\n
    - **pairs** (``list``): array that contains arrays of two str each containing coupled LNAs names;\n
    - **my_dict** (``dict``): dictionary with key-value pairs, where each key is a combined MPI and LNF name.
    """
    total_cost = 0
    for mpi, lnf in pairs:
        # build the key of the dictionary of the chains
        key = f"{mpi}_LNF-{lnf}"
        # If the key doesn't exist, assign an inf value
        total_cost += np.abs(my_dict.get(key, float('inf')))
    return total_cost


def generate_initial_solution(lna_1_list, lna_2_list) -> list:
    """
    Function to create an initial random solution (list).\n
    Parameters:\n
    - **lna_1_list** (``list``): list of names of the first LNAs (MPI names);\n
    - **lna_2_list** (``list``): list of names of the first LNAs (LNF names).
    """
    # Randomly select a set of LNAs
    # The selection is done over the larger batch of LNAs

    if len(lna_1_list) < len(lna_2_list):
        selected_lna_1 = lna_1_list
        selected_lna_2 = random.sample(lna_2_list, len(lna_1_list))
    elif len(lna_2_list) < len(lna_1_list):
        selected_lna_2 = lna_2_list
        selected_lna_1 = random.sample(lna_1_list, len(lna_2_list))
    else:
        selected_lna_1 = lna_1_list
        selected_lna_2 = lna_2_list

    # Pair selected LNA1 (MPI) names with selected LNA2 (LNF) names
    return list(zip(selected_lna_1, selected_lna_2))


def generate_neighbor(solution: list) -> list:
    """
    Function to generate a "neighbor" solution (list) by slightly modifying an existing solution (random swap)
    Parameters:\n
    - **solution** (``list``): current solution list of (mpi, lnf) pairs
    """
    neighbor = solution[:]
    # Get two random indexes to do the swap
    idx1, idx2 = random.sample(range(len(solution)), 2)
    # Swap two pairs
    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
    return neighbor


def simulated_annealing(mpi_list: list, lnf_list: list, my_dict: dict,
                        initial_temp: float, cooling_rate: float, num_iterations: int):
    """
    Simulated Annealing algorithm
    Parameters:\n
    - **mpi_list** (``list``): list of MPI names
    - **lnf_list** (``list``): list of LNF names
    - **my_dict** (``dict``): dictionary with combined MPI+LNF keys and their values
    - **initial_temp** (``float``): initial temperature for the algorithm
    - **cooling_rate** (``float``): rate at which the temperature decreases
    - **num_iterations** (``int``): number of iterations to perform
    """

    current_solution = generate_initial_solution(mpi_list, lnf_list)
    current_cost = calculate_cost(current_solution, my_dict)

    # Initialize the best solution
    best_solution = current_solution
    # Initialize the best cost
    best_cost = current_cost

    temp = initial_temp

    for i in range(num_iterations):
        # Generate a neighboring solution (a permutation of the current solution)
        new_solution = generate_neighbor(current_solution)
        new_cost = calculate_cost(new_solution, my_dict)

        # The new solution is better or is accepted probabilistically (simulated annealing)
        if new_cost < current_cost or random.uniform(0, 1) < math.exp((current_cost - new_cost) / temp):
            current_solution = new_solution
            current_cost = new_cost

            # Update the best solution if the new one is better
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost

        # Cool down the temperature
        temp *= cooling_rate

    return best_solution, best_cost


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


def Load_LNAs_Compliance(path: str) -> dict:
    """
        Load the compliance values of the bias configurations of the LNAs from the txt file in a given path\n
        Parameters:\n
        - **path** (``str``): path of the compliance file;
       Return\n
        dictionary with the keys:\n
        - LNA (configuration)
        - Compliance 67-90GHz
        - Compliance 90-116GHz
        - Average Noise T 67-90GHz
        - Average Noise T 90-116GHz
    """
    # Open the file and read all the lines
    with open(path, 'r') as file:
        lines = file.readlines()

    # The first line contains the names of the keys, comma-separated.
    # Also remove the # at the beginning of the line
    keys = lines[0].strip().replace('#', '').split(', ')
    # Initialize a dict with those keys
    compl_dict = {key: [] for key in keys}

    # Read all the other lines one by one and fill the dict
    for line in lines[1:]:
        values = line.strip().split('\t')
        for i in range(len(keys)):

            # Convert into floating point numbers
            if i > 0:
                values[i] = float(values[i])
            # Fill the dict
            compl_dict[keys[i]].append(values[i])

    return compl_dict


def Get_Best_Bias_Config(compl_dict: dict):
    """
        Read the compliance values saved on a dictionary and return the best bias configuration.
        Note that: the best configuration based on these numbers is confusing:
        a configuration could be worse in terms of compliance numbers but better because more stable!
    """
    # Initialize an empty dict
    best_config = {k: 0 for k in compl_dict.keys()}

    for idx, key in enumerate(compl_dict.keys()):
        # Compliance is terrible
        if float(compl_dict['Compliance 67-90GHz'][idx]) < 80. or float(compl_dict['Compliance 90-116GHz'][idx]) < 80.:
            pass

        else:
            # Total compliance evaluation
            curr_compl = compl_dict['Compliance 67-90GHz'][idx] + compl_dict['Compliance 90-116GHz'][idx]
            best_compl = best_config['Compliance 67-90GHz'] + best_config['Compliance 90-116GHz']

            # Compliance is better
            if curr_compl > best_compl:
                # The current config is the new best
                for k in compl_dict.keys():
                    best_config[k] = compl_dict[k][idx]

            # Compliance is the same
            elif curr_compl == best_compl:
                # Confront the Noise Temperatures
                curr_T = compl_dict['Average Noise T 67-90GHz'][idx] + compl_dict['Average Noise T 90-116GHz'][idx]
                best_T = best_config['Average Noise T 67-90GHz'] + best_config['Average Noise T 90-116GHz']

                # Noise T is better
                if curr_T < best_T:
                    # The current config is the new best
                    for k in compl_dict.keys():
                        best_config[k] = compl_dict[k][idx]

                # Noise T is the same
                elif curr_T == best_T:
                    # We keep the configuration with the lowest value at low frequencies
                    if best_config['Average Noise T 67-90GHz'] > compl_dict['Average Noise T 67-90GHz'][idx]:
                        # The current config is the new best
                        for k in compl_dict.keys():
                            best_config[k] = compl_dict[k][idx]
    return best_config


########################################################################################################################
########################################################################################################################
########################################################################################################################
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

    # Create a list to store names removing the .txt extension
    names = [os.path.splitext(file)[0] for file in file_names]

    # Initialize an empty list to contain dataset
    data = []
    # Check the kind of dataset to load it properly
    skiprows = 1 if ESO_dataset else 9

    # logging.debug(f"The names of the files will be now printed to video:\n{file_names}")

    for filename in file_names:
        # Load data from txt files, skipping "1" row for ESO data, "9" rows for LNF data
        d = np.loadtxt(f"{path}/{filename}", skiprows=skiprows)
        # Collect loaded data
        data.append(d)

    return names, data


def Plot_Single_Noise_Gain(data: list, names: list, ESO_dataset: bool, noise_spec: int,
                           output_plot_dir: str, show: bool):
    """
    Plot the Noise Temperature and the Gain of a set of Chains of LNAs and save a png picture of it.\n
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
        # logging.debug(names[i])
        ax1.plot(data[i][:, 0], data[i][:, NT_column], linewidth=2, color="royalblue",
                 label="Noise Temperature (corrected)")

        # ==============================================================================================================
        # Gain
        # ==============================================================================================================

        # Plotting Frequency (column 0) vs Gain (G_column)
        # logging.debug(names[i])
        ax1.plot(data[i][:, 0], data[i][:, G_column], linewidth=2, color="firebrick",
                 label="Gain")

        # =============================================================================
        # Noise Specifics
        if noise_spec == 80:
            ax1.plot([67, 90], [22, 22], color='k', linestyle='--', linewidth=2)
            ax1.text(80, 23, "Noise Specs, 80%", horizontalalignment='center', fontsize='25')
            ax1.plot([90, 116], [29, 29], color='k', linestyle='--', linewidth=2)
            ax1.text(105, 29, "Noise Specs, 80%", horizontalalignment='center', fontsize='25')
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
        ax1.set_xlabel("\nFrequency [GHz]\n", fontsize=25, color="black")
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.set_ylabel("\nNoise temperature [K]\n", fontsize=25, color="royalblue")

        # Create a twin axis on the right side for the Gain
        ax2 = ax1.twinx()
        # Set axis limits
        ax2.axis([67, 116, 0, 60])  # necessary?
        # Set axis label
        ax2.set_ylabel("\nGain [dB]\n", fontsize=25, color="firebrick")
        ax2.tick_params(axis='both', which='major', labelsize=20)

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
    Plot the Noise Temperature and the Gain of a set of Chains of LNAs and save a png picture of it.\n
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

    colors = ["royalblue", "darkorange", "limegreen", "firebrick", "purple",
              "hotpink", "tomato", "deepskyblue", "gold", "teal"]

    for i in range(0, len(data)):
        # ==================================================================================================================
        # Noise Temperature
        # ==================================================================================================================
        # Plotting Frequency (column 0) vs Noise Temperature (NT_column)
        # logging.debug(names[i])
        plt.plot(data[i][:, 0], data[i][:, NT_column], color=colors[i], linewidth=2, label=names[i])

        # ==================================================================================================================
        # Gain
        # ==================================================================================================================
        # Plotting Frequency (column 0) vs Gain (G_column)
        # logging.debug(names[i])
        plt.plot(data[i][:, 0], data[i][:, G_column], color=colors[i], linewidth=2)

    # Make legend for the plot
    plt.subplots_adjust(bottom=0.4)
    plt.legend(loc='upper center', shadow=True, fontsize='17', bbox_to_anchor=(0.5, -0.1))

    # =============================================================================
    # Noise Specifics
    if noise_spec == 80:
        ax1.plot([67, 90], [22, 22], color='k', linestyle='--', linewidth=2)
        ax1.text(80, 23, "Noise Specs, 80%", horizontalalignment='center', fontsize='25')
        ax1.plot([90, 116], [29, 29], color='k', linestyle='--', linewidth=2)
        ax1.text(105, 30, "Noise Specs, 80%", horizontalalignment='center', fontsize='25')
        ax1.tick_params(axis='both', which='major', labelsize=20)
        plot_name += f"{noise_spec}"

    elif noise_spec == 100:
        ax1.plot([67, 90], [26, 26], color='k', linestyle='-', linewidth=2)
        ax1.text(80, 27, "Noise Specs, 100%", horizontalalignment='center', fontsize='25')
        ax1.plot([90, 116], [33, 33], color='k', linestyle='-', linewidth=2)
        ax1.text(105, 34, "Noise Specs, 100%", horizontalalignment='center', fontsize='25')
        ax1.tick_params(axis='both', which='major', labelsize=20)
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
    ax1.tick_params(axis='both', which='major', labelsize=20)

    # Create a twin axis on the right side for the Gain
    ax2 = ax1.twinx()
    # Set axis limits
    ax2.axis([67, 116, 0, 60])  # necessary?
    # Set axis label
    ax2.set_ylabel("Gain [dB]", fontsize='25', color="black")
    ax2.tick_params(axis='both', which='major', labelsize=20)

    # Checking existence of the dir
    Path(output_plot_dir).mkdir(parents=True, exist_ok=True)
    # Save Figure of the plot
    plt.savefig(f'{output_plot_dir}/{plot_name}.png')
    # Close the Figure to save memory
    plt.close()

    # Show the plot on video if asked
    if show:
        plt.show()
