
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv('data/games.csv', lineterminator='\n')

df.shape

df.head().transpose()

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
sns.set_style(style='whitegrid')


df.groupby('platform')['title'].count().sort_values(ascending=False).plot.bar(figsize=(10, 6), rot=0);



df['release_date'] = pd.to_datetime(df['release_date'])
df['month'] = df['release_date'].dt.month
df['year'] = df['release_date'].dt.year



df.loc[df['developer'] == 'EA Sports, EA Vancouver', 'developer'] = 'EA Sports'
df.loc[df['developer'] == 'EA Vancouver', 'developer'] = 'EA Sports'



tmp = df.loc[(df['meta_overview'] != 'No score yet') & (df['user_overview'] != 'No user score yet')].copy()



tmp.groupby('platform')['meta_score', 'n_user_score'].mean().sort_values('meta_score', ascending=False).plot.bar(figsize=(10, 6), rot=0);



tmp[['meta_score']].plot.hist(bins=25, figsize=(12, 6));

tmp[['n_user_score']].plot.hist(bins=25, figsize=(12, 6));



fig, ax = plt.subplots(figsize=(12, 8))
ax.plot([0, 100], [0, 100])
tmp.plot.scatter(x='n_user_score', y='meta_score', ax=ax, alpha=0.25);

mo = tmp.groupby('meta_overview')['title'].count()
mo.name = 'meta_overview'
uo = tmp.groupby('user_overview')['title'].count()
uo.name = 'user_overview'
pd.concat([mo, uo], axis=1).sort_values('meta_overview', ascending=False).plot.bar(figsize=(12, 6));



df.loc[(df['year'] > 2013) & (df['year'] < 2018) & (df['platform'] != 'Switch')].groupby(['year', 'platform'])['title'].count().unstack().plot.bar(figsize=(10, 6));



df.groupby(['year', 'month'])['title'].count().groupby('month').mean().plot(figsize=(12, 6));



fig = plt.figure(figsize = (15,6))
ax = fig.add_subplot(1, 1, 1)
tmp.loc[(tmp['year'] > 2013) & (tmp['year'] < 2018), ('meta_score', 'year')].groupby('year').mean().plot(ax = ax, xticks=np.arange(2014, 2018, 1))
tmp.loc[(tmp['year'] > 2013) & (tmp['year'] < 2018), ('n_user_score', 'year')].groupby('year').mean().plot(ax = ax, xticks=np.arange(2014, 2018, 1));



tmp.loc[tmp['meta_score'] >= 85].groupby('platform')['title'].count().sort_values(ascending=False)



tmp.loc[tmp['n_user_score'] >= 85].groupby('platform')['title'].count().sort_values(ascending=False)



tmp['dif'] = tmp['meta_score'] - tmp['n_user_score']

ps4 = tmp.loc[tmp['platform'] == 'PlayStation 4']
xone = tmp.loc[tmp['platform'] == 'Xbox One']
switch = tmp.loc[tmp['platform'] == 'Switch']



ps4.sort_values('meta_score', ascending=False).reset_index(drop=True)[['title', 'meta_score', 'n_user_score', 'dif']][:20]


ps4.sort_values('n_user_score', ascending=False).reset_index(drop=True)[['title', 'n_user_score', 'meta_score', 'dif']][:20]



ps4.sort_values('dif', ascending=False)[:10][['title', 'meta_score', 'n_user_score', 'dif']].reset_index(drop=True)



ps4.sort_values('dif')[:10][['title', 'meta_score', 'n_user_score', 'dif']].reset_index(drop=True)


ps4.loc[ps4['title'] == 'AO Tennis'].transpose()



ps4_list = set(df.loc[df['platform'] == 'PlayStation 4', 'title'])
xone_list = set(df.loc[df['platform'] == 'Xbox One', 'title'])
switch_list = set(df.loc[df['platform'] == 'Switch', 'title'])



ps4_exclusives = tmp.loc[tmp['title'].isin(list(ps4_list.difference(xone_list).difference(switch_list))), ('title', 'n_user_score', 'meta_score')].sort_values('n_user_score', ascending=False).reset_index(drop=True)
len(ps4_exclusives)



ps4_exclusives[:25]


len(ps4_exclusives.loc[ps4_exclusives['n_user_score'] >= 80])



xone_exclusives = tmp.loc[tmp['title'].isin(list(xone_list.difference(ps4_list).difference(switch))), ('title', 'n_user_score', 'meta_score')].sort_values('n_user_score', ascending=False).reset_index(drop=True)
len(xone_exclusives)



xone_exclusives[:25]



len(xone_exclusives.loc[xone_exclusives['n_user_score'] >= 80])


switch_exclusives = tmp.loc[tmp['title'].isin(list(switch_list.difference(ps4_list).difference(xone_list))), ('title', 'n_user_score', 'meta_score')].sort_values('n_user_score', ascending=False).reset_index(drop=True)
len(switch_exclusives)



switch_exclusives[:25]



len(switch_exclusives.loc[switch_exclusives['n_user_score'] >= 80])



ps4.groupby('developer')['title'].count().sort_values(ascending=False)[:10]



dev = pd.concat([ps4.groupby('developer')['title'].count(), ps4.groupby('developer')['n_user_score'].mean(), ps4.groupby('developer')['meta_score'].mean()], axis=1)
dev.columns = ['count', 'n_user_score_avg', 'meta_score_avg']
dev.sort_values('count', ascending=False)[:10]



dev.loc[dev['count'] >= 4].sort_values('n_user_score_avg', ascending=False)[:10]



ps4.loc[ps4['developer'] == 'From Software', 'title'].reset_index(drop=True)



ps4.loc[ps4['developer'] == 'DONTNOD Entertainment', 'title'].reset_index(drop=True)



dev = pd.concat([ps4.groupby('developer')['dif'].mean().sort_values(ascending=False), ps4.groupby('developer')['title'].count()], axis=1)
dev.columns = ['dif_avg', 'count']
dev.loc[dev['count'] >= 4].sort_values('dif_avg', ascending=False)[:10]



ea = ps4.loc[ps4['developer'] == 'EA Sports', ['title', 'meta_score', 'n_user_score']].sort_values('meta_score', ascending=False)[:10]
ea.set_index('title').plot.bar(figsize=(14, 6));



dev.loc[dev['count'] >= 4].sort_values('dif_avg')[:10]

ps4.loc[ps4['developer'] == 'HB Studios Multimedia', 'title'].reset_index(drop=True)

ps4.loc[ps4['developer'] == 'Acquire', 'title'].reset_index(drop=True)



df['rating'].value_counts().plot.bar(figsize=(12, 6), rot=0);



qty = df.loc[(df['rating'].isnull() == False) & (df['rating'] != 'RP')].groupby('platform')['title'].count()
#((df.loc[df['rating'] != 'RP'].groupby(['platform', 'rating'])['title'].count().unstack().transpose() / qty) * 100).plot.bar(figsize=(12, 6), rot=0);
rat = (df.loc[df['rating'] != 'RP'].groupby(['platform', 'rating'])['title'].count().unstack().transpose() / qty) * 100
rat = rat[['PlayStation 4', 'Xbox One', 'Switch']]

fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(1, 1, 1)
rat.iloc[0].plot.bar(ax=ax, color='beige', edgecolor='white', label=rat.index[0]);
rat.iloc[1].plot.bar(ax=ax, bottom=rat.iloc[0], color='navajowhite', edgecolor='white', label=rat.index[1]);
rat.iloc[3].plot.bar(ax=ax, bottom=rat.iloc[:2].sum().values, color='lightsalmon', edgecolor='white', label=rat.index[3]);
rat.iloc[2].plot.bar(ax=ax, bottom=(rat.iloc[:2].sum() + rat.iloc[3]).values, color='firebrick', edgecolor='white', label=rat.index[2], rot=0);
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1);



tmp.loc[(tmp['rating'].isnull() == False) & (tmp['rating'] != 'RP')].groupby('rating')['n_user_score'].mean().sort_values(ascending=False).plot.bar(figsize=(12, 6), rot=0);



