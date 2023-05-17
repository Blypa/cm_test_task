"""File containing utility functions."""
import os
import glob

import pandas as pd


def read_cycles(data_folder: str, hours: str) -> pd.DataFrame:
    """Read and concat all cycles from a single hours folder in the output.

    Args:
        data_folder (str): Path to folder containing data.
        hours (str): Name of the folder inside the data folder, ex `50_h`

    Returns:
        pd.DataFrame: DataFrame with all cycles.
    """
    dfs = []

    target_folder =  os.path.join(data_folder, hours, 'data')

    file_list = glob.glob(os.path.join(target_folder, f'{hours}_cycle_*.csv'))

    # Sort all files to preserve the cycles order and read each cycle 
    for file in sorted(file_list, key=lambda filename: int(os.path.splitext(filename.split('_')[-1])[0])):
        dfs.append(pd.read_csv(file))
    
    return pd.concat(dfs)


def read_all_cycles(data_folder: str) -> pd.DataFrame:
    """Read all available data from the data folder

    Args:
        data_folder (str): Path to folder containing data

    Returns:
        pd.DataFrame: DataFrame with all cycles and hours concatenated.
    """
    hours = [
        item for item in os.listdir(data_folder)
        if item != '50_h1'  # 50_h1 is identical to 50_h, so we use only one of them
    ]

    dfs = []

    for hour_folder in sorted(hours, key=lambda filename: int(filename.split('_')[0])):
        dfs.append(read_cycles(data_folder, hour_folder))
    
    return pd.concat(dfs)
