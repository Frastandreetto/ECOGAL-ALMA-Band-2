# -*- coding: utf-8 -*-
"""
Created on Tuesday 21st of May 2024

Read all txt files containing data on Gain and Noise T in the selected directory.
Then create single plots for every file and a conclusive plot with all the measures of the files.
It produces the plots 3 times: the 2nd and 3rd plots have also some horizontal lines with the specifications required.

@author: Francesco Andreetto
(previous script: Pavel Yagoubov)
"""

# Modules & Libraries
import argparse
import f_LNA as f
import logging
import sys
from rich.logging import RichHandler


def main():
    # ==================================================================================================================
    # Main Information
    # Use the module logging to produce nice messages on the shell
    logging.basicConfig(level="INFO", format='%(message)s',
                        datefmt="[%X]", handlers=[RichHandler()])

    # ==================================================================================================================
    # CLI

    # Command Line used to start the pipeline
    command_line = " ".join(sys.argv)

    # Create the top-level argument parser by instantiating ArgumentParser
    # Note: the description is optional. It will appear if the help (-h) is required from the command line
    parser = argparse.ArgumentParser(prog='PROGRAM', description="ECOGAL-Band 2 Script to parse "
                                                                 "the Gain and Noise T conditions of the LNAs.")

    ####################################################################################################################
    # CLI Settings
    # path_file
    parser.add_argument("--path", action='store', type=str,
                        help='- Location of the txt files used to create plots (database)')
    # output_plot_dir
    parser.add_argument("--output_plot_dir", action='store', type=str, default="/Results",
                        help='- Directory where to save all the plots and results of the analysis')
    # report_dir
    parser.add_argument("--ESO", action='store', type=bool, default=True,
                        help='- Dataset selection: set True for ESO dataset, False for LNF dataset')

    # Call .parse_args() on the parser to get the Namespace object that contains all the user’s arguments.
    args = parser.parse_args()
    logging.info(args)

    # Dataset selection: set True for ESO dataset, False for LNF dataset
    ESO_dataset = args.ESO
    # My dataset path
    path = args.path
    # Output plot directory
    output_plot_dir = f"{path}{args.output_plot_dir}"

    # ==================================================================================================================
    # Default settings
    # Name of the plot
    plot_name = "LNAs_NT_Gain"
    # Title of the plot
    plot_title = "Noise T and Gain of the Chains"
    # Noise Specifications
    noise_spec = [0, 80, 100]
    # ==================================================================================================================

    # Load the data from the all the txt files in a given path
    logging.info("Loading the LNAs from the txt files.\n")
    names, data = f.Load_LNAs_Data(path=path, ESO_dataset=ESO_dataset)

    # Loop for Noise Specifications
    for ns in noise_spec:
        logging.info("Plotting Noise T and Gain of single configurations the LNAs.\n"
                     f"Noise specifics: {ns}.\n")
        # Plot Noise Temperature and Gain of the LNAs
        f.Plot_Single_Noise_Gain(data=data, names=names, ESO_dataset=True, noise_spec=ns,
                                 output_plot_dir=output_plot_dir, show=False)

        logging.info("Plotting Noise T and Gain of all the configurations of the LNAs.\n"
                     f"Noise specifics: {ns}.\n")
        # Plot Noise Temperature and Gain of the LNAs
        f.Plot_All_Noise_Gain(data=data, names=names, ESO_dataset=ESO_dataset, noise_spec=ns,
                              output_plot_dir=output_plot_dir, plot_name=plot_name, plot_title=plot_title, show=False)


if __name__ == '__main__':
    main()
