import os
import pickle
import pandas as pd
from helpers.vars import DATA_PATH


def create_folder(path, folder_name):
    folder_path = f'{path}/{folder_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_to_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_interventions(interventions_path='../interventions.csv'):
    # Loads intervention
    interventions_df = pd.read_csv(interventions_path)
    for col in interventions_df.columns:
        if col != "lang":
            interventions_df.loc[:, col] = pd.to_datetime(interventions_df.loc[:, col])
    interventions = {}
    for _, lang_info in interventions_df.T.to_dict().items():
        lang = lang_info['lang']
        del lang_info['lang']
        interventions[lang] = {k: t for k, t in lang_info.items() if not pd.isnull(t)}
    return interventions


def save_did_results(pd_did, name, folder='../csv/did'):
    pd_did.rename({'low': 'CI_lower', 'high': 'CI_upper', 'val': 'delta_m', 'pval': 'significant', 'std': 'std.err.'},
                  axis=1).to_csv(f'{folder}/{name}.csv', index=False)
