import datetime
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta, weekday
from matplotlib import lines
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from helpers.vars import mobility_changepoint_dict, mobility_reverted_dict, helper_langs, changepoints_wiki_mod

colorblind_tol = ['#117733', '#88CCEE', '#E69F00', '#882255']


def format_labels_thousand_comma(ax):
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    return ax

def plot_dates(ax, start, end, x, y, xticklabels=True, ls="solid", adjust=False, sci=True, color='blue'):
    start = datetime.datetime.strptime(str(start), "%Y%m%d")
    end = datetime.datetime.strptime(str(end), "%Y%m%d")

    if adjust is not False:
        start = start + relativedelta(weekday=adjust[0])
        end = end + relativedelta(weekday=adjust[1])

    mask = (x <= end) & (x >= start)

    ax.plot(x[mask], y[mask], ls=ls, color=color)

    if not xticklabels:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', labeltop=False, top=False, bottom=False)
    else:
        months_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_formatter(months_fmt)
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    if sci:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    return weekday(start.weekday()), weekday(end.weekday())


def plot_frequent_interventions(ax, interventions, lang, linewidth=1, color='black', intervention_delta=0):
    trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
    mob_int = pd.to_datetime(interventions[lang]['Mobility']) + relativedelta(days=intervention_delta)
    norm_int = pd.to_datetime(interventions[lang]['Normalcy']) + relativedelta(days=intervention_delta)

    ax.axvline(mob_int, color=color, linewidth=linewidth)
    # plt.text(mob_int, 1.02, 'M', transform=trans, ha='center', size=size)
    ax.axvline(norm_int, ls="--", color=color, linewidth=linewidth)
    # plt.text(norm_int, 1.02, 'N', transform=trans, ha='center', size=size)


def plot_language_dashed_dnd(pd_dnd, lang, column, changepoint, min_y=None, max_y=None, ax=None, color=None):
    if not ax:
        fig, ax = plt.subplots()

    df = pd_dnd[pd_dnd.lang == lang].copy()
    df['val_sig'], df['val_95'] = np.nan, np.nan
    col_i_v95, col_i_sig = list(df.columns).index('val_95'), list(df.columns).index('val_sig')

    for i in range(0, len(df) - 1):
        # while True:
        df_rowcurr, nextval = df.iloc[i], df.iloc[i + 1]['val']
        if not df_rowcurr.pval:
            df.iloc[i, col_i_v95] = df_rowcurr['val']
            df.iloc[i + 1, col_i_v95] = nextval
        else:
            df.iloc[i, col_i_sig] = df_rowcurr['val']
            df.iloc[i + 1, col_i_sig] = nextval

    df.plot(x='day', y=['val_sig'], ax=ax, xlim=(df['day'].min(), df['day'].max()), ylim=(min_y, max_y),
            label=[f'{lang}'], color=color, xticks=range(0, df['day'].max() + 11, 10))
    color = ax.get_lines()[-1].get_color()
    df.plot(ax=ax, x='day', y=['val_95'], label=[f'95%'], color=color, linestyle=(0, (3, 2)))  # loosely dashed

    fill_c95 = ax.fill_between(df['day'], df['low'], df['high'], color=color, alpha=.1)
    line_norm = ax.axvline(changepoint, color=color, ls='--')

    return ax


def plot_all_dashed_dnd(pd_dnd, column, min_y=None, max_y=None, limit_y=0.5, codes=None, figsize=(12, 2.5),
                        y_title=1.2, y_legend1=1.2, x_legend2=1.1, header=True, use_log=True,
                        fig_plt=None, ax_plt=None, ylabel=None, fs_legend=12, colors=None, country_legend=True):
    codes = list(pd_dnd.lang.unique()) if not codes else codes

    if not ax_plt:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_plt, ax_plt

    delta_day_dict = {code: (mobility_reverted_dict[code] - mobility_changepoint_dict[code]).days for code in
                      mobility_changepoint_dict}
    sig_line = ax.axhline(0, color='black')

    for i, code in enumerate(codes):
        ax = plot_language_dashed_dnd(pd_dnd, code, column, delta_day_dict[code],
                                      min_y=-limit_y if not min_y else min_y,
                                      max_y=limit_y if not max_y else max_y, ax=ax,
                                      color=None if not colors else colors[i])
        ax.set_ylabel(ylabel, fontsize=14)

    if not ax_plt:
        fig.text(0.075, 0.5, 'Coefficients', va='center', rotation='vertical', fontsize=16)

    # country legend
    l1 = None
    if country_legend:
        new_handles, labels = [], []
        for handle, label in zip(
                *ax.get_legend_handles_labels()):
            if '%' in label:
                continue
            new_handles.append(lines.Line2D([], [], marker="s", markersize=10, linewidth=0, color=handle.get_color()))
            labels.append(label)

        # line legend
        l1 = ax.legend(new_handles, labels, loc='center right', ncol=1, bbox_to_anchor=[x_legend2, 0.5], frameon=False,
                       labelspacing=1, fontsize=14)

    if header:
        line_hor, line_vert, line_95, fill_c95 = lines.Line2D([], [], color='gray'), \
                                                 lines.Line2D([], [], marker=r'$\vdots$', linestyle='None',
                                                              color='gray',
                                                              markersize=15, markeredgewidth=0.75), \
                                                 lines.Line2D([], [], color='gray', linestyle=(0, (3, 2)), linewidth=2), \
                                                 Patch(facecolor='black', edgecolor='black', alpha=0.16)
        handles, labels = [line_hor, line_95, fill_c95, line_vert], \
                          [column, f'{column} (95% CI)', 'CI-Band (95%)', 'Normality']

        # add additional legend (need to be like this for second legend
        ax.legend(handles, labels, numpoints=1, ncol=len(handles),
                  loc='upper center', bbox_to_anchor=[0.5, y_legend1], frameon=False, fontsize=fs_legend)

        if l1:
            ax.add_artist(l1)
        if not ax_plt:
            str_title = f' Windowed Diff-in-Diff: ' + (f'Log({column})' if use_log else f'{column}')
            fig.suptitle(str_title, fontsize=20, y=y_title)
    else:
        ax.legend().remove()

    return fig, ax


def plot_metric_and_did_all_categories(agg, pd_dnd, getters, column, code_cats, labels, min_ys, max_ys, interventions,
                                       did_label='Relative Change (log)',
                                       colors=colorblind_tol,  # ('tab:blue', 'tab:red', 'tab:orange', 'tab:green'),
                                       dates=((20200101, 20200930), (20190101, 20190930), (20180101, 20180930)),
                                       year_labels=('2020', '2019', '2018'), lines_params=("-", '--', ':'), sci=True,
                                       figsize=(40, 15), intervention_delta=0):
    fig = plt.figure(figsize=figsize)
    main_grid = gridspec.GridSpec(len(code_cats), 1, figure=fig, hspace=0.25)

    axes = [ax for ax in main_grid]
    first_ax, first_fig = None, None
    for c, codes in enumerate(code_cats):
        min_y, max_y = min_ys[c], max_ys[c]
        cat_label = labels[c]
        gs = gridspec.GridSpecFromSubplotSpec(len(codes), 3, hspace=0.8, wspace=0.25, subplot_spec=axes[c])

        # plot metrics
        if len(getters) == 1:
            getters = getters * len(dates)

        for i, code in enumerate(codes):
            curr_ax = fig.add_subplot(gs[i, 0])
            first_ax = curr_ax if not first_ax else first_ax

            idy = 0
            for date, line_params, get_val in zip(dates, lines_params, getters):
                x = get_val(code, agg)
                if idy == 0:
                    start, end = plot_dates(curr_ax, date[0], date[1], x.index, x.values, sci=sci, color=colors[i])
                else:
                    plot_dates(curr_ax.twiny(), date[0], date[1], x.index, x.values,
                               xticklabels=False, ls=line_params, adjust=(start, end), sci=sci, color=colors[i])
                    # curr_ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                # print(np.nanmin(x.values), np.nanmax(x.values))
                idy += 1
                min_x, max_x = np.nanmin(x.values), np.nanmax(x.values)
                middle_x = (max_x + min_x) / 2
                max_digits = len(str(int(max_x))) - 1 if column != 'Revert Rate' else 0
                round_digits = 2 if column != 'Revert Rate' else 3
                round_sci = lambda x, digits, ptscomma: np.round(x / 10 ** max_digits, ptscomma) * 10 ** max_digits

                curr_ax.yaxis.set_ticks([round_sci(min_x, max_digits, round_digits),
                                         round_sci(middle_x, max_digits, round_digits),
                                         round_sci(max_x, max_digits, round_digits)])

            if i < len(codes) - 1:
                curr_ax.set_xticklabels([])

            curr_ax.set_ylabel(code, fontsize='large', rotation=90)
            # curr_ax.yaxis.set_label_coords(-0.12, 2.5)
            # curr_ax.set_title(cat_label, loc='center', rotation=90,
            #                  x=-0.25, fontsize='xx-large')
            # curr_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            plot_frequent_interventions(curr_ax, interventions, code, linewidth=1.5, color=colors[i],
                                        intervention_delta=intervention_delta)

        did_ax = fig.add_subplot(gs[0:4, 1:])
        sub_fig, ax = plot_all_dashed_dnd(pd_dnd, column, min_y=min_y, max_y=max_y, codes=codes, header=False,
                                          fig_plt=fig, ax_plt=did_ax, fs_legend='x-large', colors=colors,
                                          country_legend=False, y_legend1=1.3)

        first_fig = sub_fig if not first_fig else first_fig

        # f'{column} for Large Wikis'
        ax.set_ylabel(did_label, fontsize='x-large')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # right_plot plots
        lang_handles, lang_labels = [], []
        for color, code in zip(colors, codes):
            lang_handles.append(Line2D([], [], marker="s", markersize=10, linewidth=0, color=color))
            lang_labels.append(code)

        # country line legend
        l1 = ax.legend(lang_handles, [f'{helper_langs[lang]} ({lang})' for lang in lang_labels],
                       loc='center left', ncol=1, frameon=False, bbox_to_anchor=(1, 0.5),
                       labelspacing=1, fontsize='large')

        if c == 0:
            line_hor, line_vert, line_95 = lines.Line2D([], [], color='gray', linewidth=2), \
                                           lines.Line2D([], [], marker=r'$\vdots$', linestyle='None',
                                                        color='gray',
                                                        markersize=15, markeredgewidth=0.75), \
                                           lines.Line2D([], [], color='gray', linestyle=(0, (3, 2)),
                                                        linewidth=2)
            leg_handels, leg_labels = [(line_hor, line_95), line_vert], \
                                      [f'Significance (solid) or non-significance (dashed)', 'Normality']

            # add additional legend (need to be like this for second legend
            l2 = ax.legend(leg_handels, leg_labels, numpoints=1, ncol=len(leg_handels),
                           loc='upper center', bbox_to_anchor=[0.5, 1.275], frameon=False, fontsize='large',
                           handler_map={tuple: HandlerTuple(ndivide=None, pad=1)}, handlelength=6)
            ax.add_artist(l1)

        ax.set_xlabel(None)

    # left lines
    lines_years = [Line2D([], [], color="gray", lw=1.5, ls=l) for l in lines_params]
    leg_years = first_ax.legend(
        handles=lines_years, labels=year_labels, loc='upper center', bbox_to_anchor=(0.5, 2.5),
        ncol=3, fancybox=False, shadow=False,
        frameon=False, edgecolor=None, fontsize='large'
    )

    curr_ax.set_xlabel('Date\n\n(a)', labelpad=15, fontsize='x-large')

    # right_plots
    lines_restrictions = [
        Line2D([], [], marker=r'$\vert$', color="gray", markersize=15, markeredgewidth=0.3, ls='None'),
        Line2D([], [], marker=r'$\vdots$', color="gray", markersize=15, markeredgewidth=0.3, ls='None')]

    leg_rest = first_ax.legend(
        handles=lines_restrictions, labels=['Mobility', 'Normality'], loc='upper center',
        bbox_to_anchor=(0.5, 3.3),
        ncol=2, fancybox=False, shadow=False,
        frameon=False, edgecolor=None, fontsize='large'
    )
    first_ax.add_artist(leg_years)
    ax.set_xlabel('Day after Mobility Changepoint\n\n(b)', labelpad=15, fontsize='x-large')
    fig.suptitle(column, fontsize='xx-large', x=0.04, y=0.5, ha='left', va='center', rotation=90)
    fig.text(0.07, 0.5, labels[1], ha='left', va='center', rotation=90, fontsize='x-large')
    fig.text(0.07, 0.225, labels[2], ha='left', va='center', rotation=90, fontsize='x-large')
    fig.text(0.07, 0.775, labels[0], ha='left', va='center', rotation=90, fontsize='x-large')
    return fig


def plot_covid_ratio(dicts_covidratio, large_wikis, medium_wikis, small_wikis):
    figFull, axFull = plt.subplots(3, 2, figsize=(12, 6))
    axLarge = axFull[0, :]
    axMedium = axFull[1, :]
    axSmall = axFull[2, :]

    for i, code in enumerate(large_wikis + medium_wikis + small_wikis):
        merged = dicts_covidratio[code]
        merged[0]['date_dt'] = pd.to_datetime(merged[0]['date_str'])
        print(f'======== {code}:')
        print(f'Max Edit ratio aimed towards Covid articles: ', merged[1].ratio_covid_edits)
        print(f'Max ratio of edited articles which are Covid: ', merged[1].ratio_covid_articles)

        if code in large_wikis:
            ax = axLarge
        elif code in medium_wikis:
            ax = axMedium
        else:
            ax = axSmall

        merged[0].fillna(0).plot(x='date_dt', y=['ratio_covid_edits'],
                                 title=f'Percentage of edits aimed toward COVID-19 Articles', rot=0, ax=ax[0],
                                 label=[helper_langs[code]], alpha=.8, color=colorblind_tol[i % 4])
        merged[0].fillna(0).plot(x='date_dt', y=['ratio_covid_articles'],
                                 title=f'Percentage of edited articles that are COVID-19 articles', rot=0, ax=ax[1],
                                 label=[helper_langs[code]], alpha=.8, color=colorblind_tol[i % 4])

        ax[0].yaxis.set_major_formatter(ticker.PercentFormatter(1.0, 1))
        ax[1].yaxis.set_major_formatter(ticker.PercentFormatter(1.0, 1))
        ax[0].legend().set_visible(False)
        ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

        if ax is not axLarge:
            ax[0].set_title('')
            ax[1].set_title('')

        if ax is not axSmall:
            ax[0].set_xlabel('')
            ax[1].set_xlabel('')
            plt.setp(ax[1].get_xticklabels(), visible=False)
            plt.setp(ax[0].get_xticklabels(), visible=False)

    axLarge[0].set_ylabel('Large\nWikipedias', fontsize='large')
    axMedium[0].set_ylabel('Medium\nWikipedias', fontsize='large')
    axSmall[0].set_ylabel('Small\nWikipedias', fontsize='large')
    axSmall[0].set_xlabel('Date\n\n(a)', fontsize='large')
    axSmall[1].set_xlabel('Date\n\n(b)', fontsize='large')

    return figFull


def plot_main_intro_figure(agg, code, col='count', user_kinds=('account', 'anonymous'), years_merge=False,
                           from_date='2020-01-01', to_date='2020-10-07', include_covid=None, window=7, figsize=(12, 5),
                           ylabel='Edits in the English Wikipedia', xlabel='Date'):
    user_kinds = list(user_kinds)
    # calc csum
    n_days = (datetime.datetime.strptime(to_date, "%Y-%m-%d") - datetime.datetime.strptime(from_date, "%Y-%m-%d")).days

    # extract and aggregate data
    df_code = agg[agg.code == code][['date', col, 'covid', 'user_kind']]
    df_code = pd.concat([df_code[(pd.to_datetime(df_code.date) >= pd.to_datetime(f'{year}-01-01')) & (
            pd.to_datetime(df_code.date) < (pd.to_datetime(f'{year}-01-01') + relativedelta(days=n_days)))] for
                         year in ([2019, 2020] if not years_merge else [2018, 2019, 2020])])
    df_code = df_code if not include_covid else df_code[df_code.covid == include_covid]
    df_code = df_code[df_code.user_kind.isin(user_kinds)].groupby(['date'])['count'].sum().reset_index()
    df_code['date'] = pd.to_datetime(df_code.date)
    df_code['year'] = pd.to_datetime(df_code.date).dt.year
    df_code['day'] = pd.to_datetime(df_code.date).dt.dayofyear

    # combine 2018 and 2019
    if years_merge:
        merged = df_code[df_code.year < 2020].groupby('day')['count'].mean()
        df_code = df_code[df_code.year >= 2019].copy()
        df_code.loc[df_code.year == 2019, 'count'] = merged.values

    # build plot dataframe by pivoting
    styles = ['--', '-']
    df_pvt = df_code.pivot(index='day', columns='year')

    # now, take care of windowing
    df_pvt_years = df_pvt[df_pvt.columns[:2]]
    df_pvt = df_pvt.rolling(window=window, center=True).mean()
    df_pvt = pd.concat([df_pvt_years, df_pvt], axis=1)
    df_pvt['diff'] = df_pvt[('count', 2020)] - df_pvt[('count', 2019)]

    # === PLOTTING
    # plots figure for english edits and csum of edits
    fig, ax1 = plt.subplots(figsize=figsize)
    c_l, c_fill_neg, c_fill_pos, c_events = 'tab:blue', 'tab:red', 'tab:green', 'tab:gray'

    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    df_pvt.plot(x=('date', 2020), y=col, color=c_l, lw=2, style=styles, ax=ax1)

    x, y1, y2, alpha_fill = df_pvt[('date', 2020)].values, df_pvt[('count', 2019)], df_pvt[('count', 2020)], 0.7
    ax1.fill_between(x, y2, y1, where=y1 >= y2, facecolor=c_fill_neg, interpolate=True, alpha=alpha_fill)
    ax1.fill_between(x, y2, y1, where=y1 < y2, facecolor=c_fill_pos, interpolate=True, alpha=alpha_fill)

    ax1.set_xlabel(xlabel, fontsize='large')
    ax1.set_ylabel(ylabel, fontsize='large')
    format_labels_thousand_comma(ax1)

    # add legend info
    lines = [Line2D([], [], color=c_l, lw=2, ls=styles[1]), Line2D([], [], color=c_l, lw=2, ls=styles[0])]
    fill_pos, fill_neg = Patch(facecolor=c_fill_pos, edgecolor=c_fill_pos, lw=1, alpha=alpha_fill), \
                         Patch(facecolor=c_fill_neg, edgecolor=c_fill_neg, lw=1, alpha=alpha_fill)

    label_2019 = '2019' if not years_merge else 'Mean of 2019 and 2018'
    ax1.legend(handles=lines + [(fill_pos, fill_neg,)],
               labels=['2020', label_2019, 'Surplus (green) or deficit (red) in edits in 2020'],
               loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3,
               fancybox=False, shadow=False, frameon=False, edgecolor=None, handlelength=2.5,
               handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize='large')

    # plot events
    trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)

    # for now, very static.
    for changepoint, date in changepoints_wiki_mod.items():
        day = pd.to_datetime(date)
        coordinates = day - relativedelta(days=5)
        ax1.axvline(day, alpha=0.5, color=c_events, lw=1)
        ax1.text(coordinates, 0.025, day.strftime('%b %d'), color=c_events, rotation=90, va='bottom', transform=trans)
        ax1.text(coordinates, 1.015, changepoint, color=c_events, rotation=45, va='bottom', ha='left', transform=trans)

    # === Arrows
    c_arrow = 'black'  # c_events
    max_day = df_pvt.loc[df_pvt['diff'].idxmax()]
    max_date = max_day['date', 2020]
    from_val, to_val = max_day['count', 2019], max_day['count', 2020]
    ax1.text(max_date + relativedelta(days=2), 0.2825,
             f"{max_date.strftime('%b %d')}: +{round(max_day['diff'].values[0]):,} Edits", color=c_arrow, rotation=90,
             va='bottom', ha='left', transform=trans)
    arr = plt.arrow(max_date, from_val + 2000, 0, to_val - from_val - 3500, width=0.5,
                    length_includes_head=True, head_length=1500, head_width=3, linewidth=0, shape='full',
                    color=c_arrow)
    arr2 = plt.arrow(max_date, to_val - 2500, 0, from_val - to_val + 3500, width=0.5,
                     length_includes_head=True, head_length=1500, head_width=3, linewidth=0, shape='full',
                     color=c_arrow)
    ax1.add_patch(arr)
    ax1.add_patch(arr2)

    return fig
