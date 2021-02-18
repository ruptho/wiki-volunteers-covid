from collections import defaultdict
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from dateutil.relativedelta import relativedelta

from helpers.vars import mobility_changepoint_dict


def replace_outliers_aggdict(agg_dict, col, user_kinds=('anonymous', 'account'), n_std_dev=5, comp_col='median',
                             replace_col='median', dev_func='mad', covid=None):
    new_dict = {}
    for code, df_code in agg_dict.items():
        df_code = df_code[df_code.covid == covid] if covid is not None else df_code
        df_day = df_code[df_code.user_kind.isin(user_kinds)].groupby('date')[col].sum().reset_index()
        df_day.date = pd.to_datetime(df_day.date)
        df_day['month'] = df_day.date.apply(lambda d: f'{d.year}-{d.month:02d}')
        monthly = df_day.groupby(['month'])[col].agg(['median', 'mean', 'sum', 'std', 'mad'])
        df_day = df_day.join(monthly, on='month', how='inner')

        outliers = (df_day[col] - df_day[comp_col]).abs() > (df_day[dev_func] * n_std_dev)
        # print(code, len(df_day[outliers]), df_day[outliers][['date', col, comp_col, dev_func]])
        df_day.loc[outliers, col] = np.nan
        df_day[col].fillna(df_day[outliers][replace_col], inplace=True)
        new_dict[code] = df_day[['date', col]]
        new_dict[code].set_index('date', inplace=True)
    return new_dict


def replace_outliers_in_series(df_code, col, n_std_dev=5, comp_col='median', replace_col='median', dev_func='mad'):
    df_day = df_code.reset_index()
    df_day.date = pd.to_datetime(df_day.date)
    df_day['month'] = df_day.date.apply(lambda d: f'{d.year}-{d.month:02d}')
    monthly = df_day.groupby(['month'])[col].agg(['median', 'mean', 'sum', 'std', 'mad'])
    df_day = df_day.join(monthly, on='month', how='inner')

    outliers = (df_day[col] - df_day[comp_col]).abs() > (df_day[dev_func] * n_std_dev)
    df_day.loc[outliers, col] = np.nan
    df_day.loc[outliers, col].fillna(df_day[outliers][replace_col], inplace=True)
    df_day = df_day[['date', col]]
    df_day.set_index('date', inplace=True)
    return df_day.squeeze()


def get_standard_error_sum(results, covariates):
    # 95CI is approximated with +- 2 sum_variance_standard_erro
    # get the variance covariance matrix
    # print(covariates)
    # see, for example: https://stats.stackexchange.com/a/3657
    # this is what this does!
    # Note: diagonal of cov = var, se² = var.
    vcov = results.cov_params().loc[covariates, covariates].values

    # calculate the sum of all pair wise covariances by summing up off-diagonal entries
    off_dia_sum = np.sum(vcov)
    # variance of a sum of variables is the square root
    return np.sqrt(off_dia_sum)


def extract_interaction_coefficient(res, coefficient, lang_col, lang, is_baseline):
    # baseline => this is basically how it is realized here in the code. Since the overall result for the language
    # ... has to depend on "something" -> (increase from 0). This baseline is the first language in the codes
    # therefore, we have to grab the "pos_col":"treated_col", which basically equals the coefficient of
    # '{treated_col}:{pos_col}:{baseline_lang}". This is basically how it is technically implemented within
    # statsmodels
    # - beta_0 (intercept): danish wiki (language=0), year 2018/2019 (=0), pre-changepoint period (=0)
    # - beta_1 (language) - level change that given language  (categorical) introduces in comparison to the baseline
    #   language in 2018/2019 (=0, "basically intercept")
    # - beta_2 (year) - overall level change from 2018/2019 (=0) in comparison to 2020 (=1) for the baseline wiki
    # - beta_3 (period): seasonal effect for baseline language (pre-to-post over both years)
    # - beta_4 (year:language): How did overall level change from 2018/2019 to 2020 for the given language in comparison
    #   to the baseline wiki (interaction). "What were the language-specific effects for the year change"
    # - beta_5 (period:language): Seasonal effect for language in comparison to baseline language (pre-to-post)
    # - beta_6 (year:period): The change in V from pre- to post-changepoint date over all years for the baseline wiki,
    #   after change in year (year_dummy) or period alone (which is captured by period_dummy).
    #    + What is the additional effect in 2020 for this period of time, that was not pre-existing in 2019?
    # - beta_7 (year:period:language): the change in V after the changepoint in 2020 in comparison to the baseline wiki,
    #   after change in year (year_dummy) or period alone (which is captured by period_dummy)

    # relative starting point for language = intercept + language = beta_0 + beta_1
    #   (what is the intercept for the given language model here, basically)
    # relative seasonal effect for non-baseline language (over both years) = period + period:language = beta_3 + beta_5
    # relative yearly effect for non-baseline language = year + year_language = beta_2 + beta_4
    # relative change from pre- to- post-changepoint levels in 2020 for non-baseline language =
    #   year:period + year:period:language = beta_6 + beta_7

    # The categorical encoding of using n-1 (for n categories) predictors is necessary, because otherwise we would
    # introduce multicollinearity ("the dummy trap" -
    # e.g., https://en.wikipedia.org/wiki/Dummy_variable_(statistics)#Incorporating_a_dummy_independent,
    # http://facweb.cs.depaul.edu/sjost/csc423/documents/dummy-variable-trap.htm)
    if is_baseline:
        if coefficient == lang_col:
            coefficient = 'Intercept'  # see notes above
        val = res.params[coefficient]
        std = get_standard_error_sum(res, [coefficient])
    else:
        if coefficient == lang_col:
            # note that the lang parameter has to be interpreted slightly differently
            val = res.params['Intercept'] + res.params[f'{lang_col}[T.{lang}]']
            std = get_standard_error_sum(res, ['Intercept', f'{lang_col}[T.{lang}]'])
        else:
            val = res.params[coefficient] + res.params[f'{coefficient}:{lang_col}[T.{lang}]']
            std = get_standard_error_sum(res, [coefficient, f'{coefficient}:{lang_col}[T.{lang}]'])
    return val, std


def get_diffs_in_diffs_result(df, codes, value_col='value_log', pos_col='period_dummy',
                              treated_col='year_dummy', lang_col='lang', z=2):
    res_ = smf.ols(formula=f'{value_col} ~ {treated_col} * {pos_col} * {lang_col}', data=df).fit()
    res = res_.get_robustcov_results(cov_type='HC0')
    res = statsmodels.regression.linear_model.RegressionResultsWrapper(res)

    # print("R2: {}".format(res.rsquared))

    # print(res.summary())
    # we assume codes is ordered here (that's how smf.ols does it) -> baseline is first element alphabetically
    baseline, codes_in_df = None, sorted(df[lang_col].unique())
    df_lists = defaultdict(list)

    for lang in sorted(codes):
        if lang not in codes_in_df:
            # print(f'Skipped {lang}.')
            continue

        if not baseline:  # first lang = baseline
            baseline = lang

        for coeff in [f'{lang_col}', f'{treated_col}', f'{pos_col}', f'{treated_col}:{pos_col}']:
            val, std = extract_interaction_coefficient(res, coeff, lang_col, lang, lang == baseline)

            # Conf_int from std-dev: https://stats.stackexchange.com/a/303209
            tmp_dict = {
                "lang": lang,
                "low": val - z * std,
                "high": val + z * std,
                "val": val,
                "pval": (val - z * std > 0) or (val + z * std < 0),
                "std": std
            }

            df_lists[coeff].append(tmp_dict)

    return {coeff: pd.DataFrame(df_list) for coeff, df_list in df_lists.items()}  # , res


def build_diff_in_diff_time(agg, codes, interventions_pre, interventions_pos, time_int_before, time_int_after, column,
                            user_kinds=('account', 'anonymous'), use_log=True, agg_func='sum', add_rel=False,
                            delta_day_dict_after=None, z_val=2, covid=None, extra_control_year=False,
                            n_std_outliers=None, extract_all_coefficients=False):
    df_list = []
    for lang in codes:  # + [code +".m" for code in codes]:
        time_int_after = delta_day_dict_after[lang] if delta_day_dict_after else time_int_after
        agg_code = agg[lang].copy()
        agg_code = agg_code[agg_code.covid] if covid is True else agg_code[
            ~agg_code.covid] if covid is False else agg_code
        # print(agg_code.head())
        y = agg_code[agg_code['user_kind'].isin(user_kinds)].groupby(['date'])[column].agg(agg_func)
        if n_std_outliers:
            y = replace_outliers_in_series(y, column, n_std_outliers)

        x = y.index
        names = [("pre_treat", "year_treat"), ("pos_treat", "year_treat"), ("pre_treat", "year_control"),
                 ("pos_treat", "year_control")]

        if lang not in interventions_pre or lang not in interventions_pos:
            continue

        intervention_pre, intervention_pos = interventions_pre[lang], interventions_pos[lang]
        # think of leapdays for relativedelta here!
        if add_rel:
            intervention_pre = intervention_pre + relativedelta(days=time_int_before)
        pre_treated = [intervention_pre - relativedelta(days=time_int_before), intervention_pre]
        pos_treated = [intervention_pos, intervention_pos + relativedelta(days=time_int_after)]

        # watch out for leapyears here! had to learn this the hard way.
        pre_control = [pre_treated[0] + relativedelta(leapdays=1) - relativedelta(years=1),
                       pre_treated[1] + relativedelta(leapdays=1) - relativedelta(years=1)]
        pos_control = [pos_treated[0] + relativedelta(leapdays=1) - relativedelta(years=1),
                       pos_treated[1] + relativedelta(leapdays=1) - relativedelta(years=1)]

        # include additional year back for sanity check
        pre_control_2, pos_control_2 = None, None
        if extra_control_year:
            pre_control_2 = [pre_treated[0] + relativedelta(leapdays=1) - relativedelta(years=2),
                             pre_treated[1] + relativedelta(leapdays=1) - relativedelta(years=2)]
            pos_control_2 = [pos_treated[0] + relativedelta(leapdays=1) - relativedelta(years=2),
                             pos_treated[1] + relativedelta(leapdays=1) - relativedelta(years=2)]
            names += [("pre_treat", "year_control"), ("pos_treat", "year_control")]

        for idx, dates in enumerate([pre_treated, pos_treated, pre_control, pos_control, pre_control_2, pos_control_2]):
            if dates is None:
                continue

            mask = (x >= pd.to_datetime(dates[0])) & \
                   (x < pd.to_datetime(dates[1]))

            for mean_i, mean_value in enumerate(y[mask].values):
                # avoid 0 errors here!
                if mean_value <= 0:
                    # handles this by just taking the previous value, for now
                    # (if first index, take next value.)
                    mean_value = y[mask].values[mean_i - 1] if mean_i > 0 else y[mask].values[mean_i + 1]
                    if mean_value == 0:  # fallback fallback. only needed for active editors usually.
                        mean_value = 1

                # print(idx, names[idx][0], names[idx][1])
                df_list.append({
                    "lang": lang,
                    "period": names[idx][0],
                    "group": names[idx][1],
                    "value_log": np.log(mean_value),
                    "value": mean_value,
                    "date": idx  # don't need this
                })

    df_dnd = pd.DataFrame(df_list)
    # note that this two names should probably be switched here. think hard before renaming anything - don't break!

    df_dnd["period_dummy"] = (df_dnd.period == "pos_treat").astype(int)
    df_dnd["year_dummy"] = (df_dnd.group == "year_treat").astype(int)
    df_dnd_results = get_diffs_in_diffs_result(df_dnd, codes, value_col='value_log' if use_log else 'value', z=z_val)
    if not extract_all_coefficients:
        df_dnd_results = df_dnd_results[f'year_dummy:period_dummy']
    return df_dnd, df_dnd_results,  # did_table_results


def build_windowed_diff_in_diff(agg, codes, column, time_int_before=30, time_int_window=7,
                                user_kinds=('account', 'anonymous'), use_log=True, z_val=2, day_range=120,
                                covid=None, extra_control=False, days_before=0, n_std_outliers=None,
                                extract_all_coefficients=False, window_delta=0):
    # window_delta is used for control experiments
    mobility = {code: mob_date + relativedelta(days=window_delta) for code, mob_date in
                mobility_changepoint_dict.items()}

    pd_diffs = [] if not extract_all_coefficients else defaultdict(list)
    # pd_diff_setup_list = []
    for day in range(-days_before, day_range):
        window_mobility_dict = {code: mob_date + relativedelta(days=day) for code, mob_date in mobility.items()}
        df_diff, df_diff_res = build_diff_in_diff_time(agg, codes, mobility, window_mobility_dict, time_int_before,
                                                       time_int_window, column, user_kinds, use_log, z_val=z_val,
                                                       covid=covid, extra_control_year=extra_control,
                                                       n_std_outliers=n_std_outliers,
                                                       extract_all_coefficients=extract_all_coefficients)
        # pd_diff_setup_list.append(df_diff)
        if not extract_all_coefficients:
            df_diff_res['window'] = [window_mobility_dict[c] for c in df_diff_res.lang]
            df_diff_res['day'] = day
            pd_diffs.append(df_diff_res)
        else:
            for coeff, df_res in df_diff_res.items():
                df_res['window'] = [window_mobility_dict[c] for c in df_res.lang]
                df_res['day'] = day
                pd_diffs[coeff].append(df_res)

    return pd.concat(pd_diffs) if not extract_all_coefficients \
        else {coeff: pd.concat(df) for coeff, df in pd_diffs.items()}  # , pd_diff_setup_list


def compute_covidratio(dict_edits_bytitle, date_from=20200101, date_to=20200931, covid_col='covid'):
    dict_covidratio = {}
    for code, df_code in dict_edits_bytitle.items():
        df_edits_covid = df_code[df_code[covid_col]]
        df_edits_nocovid = df_code[~df_code[covid_col]]

        grouped_covid = df_edits_covid[(df_edits_covid.date >= date_from) & (df_edits_covid.date < date_to)].groupby(
            ['date']).agg({'event_user_id': ['sum', 'size']})
        grouped_covid.columns = grouped_covid.columns.droplevel()
        grouped_covid = grouped_covid.rename({'sum': 'covid_edits', 'size': 'covid_articles'}, axis=1)
        grouped_noncovid = \
            df_edits_nocovid[(df_edits_nocovid.date >= date_from) & (df_edits_nocovid.date < date_to)].groupby(
                ['date']).agg({'event_user_id': ['sum', 'size']}, axis=1)
        grouped_noncovid.columns = grouped_noncovid.columns.droplevel()
        grouped_noncovid = grouped_noncovid.rename({'sum': 'noncovid_edits', 'size': 'noncovid_articles'}, axis=1)

        merged = pd.concat([grouped_covid, grouped_noncovid], axis=1)
        merged['ratio_covid_edits'] = merged.apply(lambda row: row.covid_edits / (row.covid_edits + row.noncovid_edits),
                                                   axis=1)
        merged['ratio_covid_articles'] = merged.apply(
            lambda row: row.covid_articles / (row.covid_articles + row.noncovid_articles),
            axis=1)
        merged['date_str'] = merged.index.astype(str)
        # merged.plot(x='date_str', y=['ratio_covid_edits'], title=code, rot=40)
        dict_covidratio[code] = (merged, merged.loc[merged['ratio_covid_edits'].idxmax()],
                                 merged.loc[merged['ratio_covid_articles'].idxmax()])
    return dict_covidratio
