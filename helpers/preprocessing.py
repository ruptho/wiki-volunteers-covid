import time
import traceback
import pandas as pd

from helpers.logger import Logger


def process_edits(dict_date, code):
    print('reading and making sure all dates are filled')
    df_code = dict_date[code].reset_index()
    df_code['date'] = pd.to_datetime(df_code.date.astype(str), format='%Y%m%d')
    df_gb = df_code.drop(['page_title', 'page_title_historical_norm'], axis=1).groupby(
        [pd.Grouper(freq="d", key="date"), "user_kind", "page_title_norm", 'covid']).sum()
    df_gb = df_gb.reset_index().rename({'page_title_norm': 'title'}, axis=1)
    return df_gb


def process_newcomers(dict_authors, code, final):
    df_author = dict_authors[code].reset_index()
    df_author["date"] = pd.to_datetime(df_author.date.astype(str), format='%Y%m%d')
    df_author['user_kind'] = 'account'
    df_author = df_author.rename(columns={'event_user_id': 'actor_user'})

    final = final.merge(df_author, left_on=["date", "user_kind"], right_on=["date", "user_kind"], how="left")
    return final


def process_reverts(dict_reverts, code, final):
    df_rev = dict_reverts[code].reset_index()
    df_rev["date"] = pd.to_datetime(df_rev.date.astype(str), format='%Y%m%d')
    final = final.merge(df_rev, left_on=["date", "user_kind"], right_on=["date", "user_kind"], how="left")
    return final


def aggregate_preprocess_results(codes, dict_edits, dict_newcomers, dict_reverts):
    # df_topics, topics = load_topics(path_topics)
    aggs = []

    for code in codes:
        start = time.time()
        try:
            df_gb = process_edits(dict_edits, code)
            # group edits
            df_gb.rename({"title": "index", 'event_user_id': 'count', 'revision_text_bytes': 'rev_len_sum'},
                         inplace=True, axis=1)
            final = df_gb.groupby(["date", "covid", "user_kind"]).sum().reset_index()

            final = process_newcomers(dict_newcomers, code, final)
            final = process_reverts(dict_reverts, code, final)

            final = final.fillna(0)
            final["code"] = code
            aggs.append(final.loc[:, final.columns != 'index'])
        except Exception as e:
            traceback.print_exc()
            Logger.instance('pipeline').info(f'Error for {code}: {str(e)}')
        Logger.instance('pipeline').info(f'Processing {code} took {time.time() - start}')
    final_aggs = pd.concat(aggs)

    return final_aggs
