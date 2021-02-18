import pandas as pd

# Path to data. change accordingly
DATA_PATH = f'../data'
DUMPS_PATH = f'{DATA_PATH}/dumps'
PRE_PATH = f'{DATA_PATH}/preprocessing'
RES_PATH = f'{DATA_PATH}/results'

# Codes of Wikis to download an analyze
CODES = ["de", "fr", "it", "sr", "no", "ko", "da", "sv", "ja", "nl", "fi", "en"]

helper_langs = {"sv": "Swedish", "de": "German", "fr": "French", "it": "Italian", "sr": "Serbian", "no": "Norwegian",
                "ko": "Korean", "da": "Danish", "ja": "Japanese", "nl": "Dutch", "fi": "Finnish", "en": "English"}

# Changepoints as extracted via the methodology by Horta Ribeiro et al. (2020) -
# see https://arxiv.org/abs/2005.08505
wiki_code = ['da', 'fi', 'de', 'it', 'ja', 'nl', 'no', 'ko', 'sv', 'en', 'fr', 'sr']

mobility_changepoints = ['2020-03-11', '2020-03-16', '2020-03-16', '2020-03-11', '2020-03-31',
                         '2020-03-16', '2020-03-11', '2020-02-25', '2020-03-11', '2020-03-16',
                         '2020-03-16', '2020-03-16']

mobility_changepoints = pd.to_datetime(mobility_changepoints)

mobility_reverted = ['2020-06-05', '2020-05-21', '2020-07-10', '2020-06-26', '2020-06-14',
                     '2020-05-29', '2020-06-04', '2020-04-15', '2020-06-05', '2020-05-21',
                     '2020-07-02', '2020-05-02']

mobility_reverted = pd.to_datetime(mobility_reverted)

mobility_changepoint_dict = dict(zip(wiki_code, list(mobility_changepoints)))
mobility_reverted_dict = dict(zip(wiki_code, list(mobility_reverted)))

# Changepoints for main figure
# extracted from: https://wikimediafoundation.org/covid19/data/
changepoints_wiki_mod = {
    '1st Death in China': '2020-01-11',
    'Disease Named "COVID-19"': '2020-02-11',
    'Lockdown in Italy': '2020-03-09',
    '1 Million Cases': '2020-04-02',
    '200,000 Deaths': '2020-04-26',
    '5 Million Cases': '2020-05-21',
    '16 Million Cases': '2020-07-26',
    '1 Million Deaths': '2020-09-28'}
