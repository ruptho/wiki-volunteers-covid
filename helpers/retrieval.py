import pandas as pd
import requests as rq

WM_API = 'https://wikimedia.org/api/rest_v1'

EDITOR_ACT_ALL = 'all-activity-levels'
EDITOR_ACT_1_4 = '1..4-edits'
EDITOR_ACT_5_24 = '5..24-edits'
EDITOR_ACT_25_99 = '25..99-edits'
EDITOR_ACT_100 = '100..-edits'
EDITOR_ALL_ACTIVITY_LEVELS = [EDITOR_ACT_ALL, EDITOR_ACT_1_4, EDITOR_ACT_5_24, EDITOR_ACT_25_99, EDITOR_ACT_100]


def retrieve_active_editors(lang, activity_level=EDITOR_ACT_ALL, start=20180101, end=20201201,
                            granularity='daily', editor_type='user', page_type='content'):
    response = rq.get(f'{WM_API}/metrics/editors/aggregate/{lang}.wikipedia.org/{editor_type}/{page_type}/'
                      f'{activity_level}/{granularity}/{start}/{end}')
    data = response.json()
    df_act = pd.DataFrame(data['items'][0]['results'])
    df_act['date'] = pd.to_datetime(df_act.timestamp).dt.date
    df_act[activity_level] = df_act.editors
    return df_act[['date', activity_level]]


def retrieve_all_editor_activity_levels(lang, start=20180101, end=20201201, granularity='daily', editor_type='user',
                                        page_type='content'):
    pd_list = []
    for activity_level in EDITOR_ALL_ACTIVITY_LEVELS:
        pd_list.append(retrieve_active_editors(lang, activity_level, start, end, granularity, editor_type, page_type))

    pd_list = [df.set_index('date', drop=True) for df in pd_list]
    pd_merged = pd.concat(pd_list, axis=1)
    pd_merged['code'] = lang
    return pd_merged


def retrieve_all_editor_activity_levels_for_all_wikis(codes, start=20171201, end=20201201, granularity='daily',
                                                      editor_type='user', page_type='content'):
    pd_list = [
        retrieve_all_editor_activity_levels(code, start, end, granularity, editor_type, page_type) for code in codes]
    return pd.concat(pd_list, axis=0)
