
import numpy as np, pandas as pd
from datetime import datetime
from langdetect import detect

"""At this moment we have 3 .csv files for EACH platform:
<br/>1- Games information
<br/>2- Critics reviews
<br/>3- Users reviews
<br/>Total: 9 files
<br/><br/>The objective is to have just 2 cleaned .csv files: 
<br/>1- Games information of all platforms
<br/>2- Reviews (critics + users) of all platforms

## Games info

#### PS4

We first need to load the .csv file
"""

df = pd.read_csv('data/ps4_games.csv', lineterminator='\n')



df.head().transpose()



df.info()


def obj_to_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col].isnull(), col] = 0

obj_to_numeric(df, ['user_score', 'user_pos', 'user_mixed', 'user_neg'])

"""Fill meta_overview and user_overview null values with their corresponding categories when there's no score."""

df.loc[df['meta_overview'].isnull(), 'meta_overview'] = 'No score yet'
df.loc[df['user_overview'].isnull(), 'user_overview'] = 'No user score yet'

df.info()

"""Let's see meta_score and user_score min and max values. We are looking for outliers."""

df.describe().loc[['min', 'max'], ['meta_score', 'user_score']]

"""No outliers but the columns are in different range. They should be in the same range in order to be able to compare them."""

df['n_user_score'] = df['user_score'] * 10

"""release_date column contain strings, for instance "Jul 19, 2016". We must transform those strings into datetime."""

df['release_date'] = df['release_date'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%b %d, %Y')))

"""Save cleaned dataframe"""

df.to_csv('ps4_games_cleaned.csv', index=False, encoding = 'utf-8')

"""#### Xbox One

Apply the same steps for Xbox One and Switch game information files
"""

df = pd.read_csv('xboxone_games.csv', lineterminator='\n')

df.info()

obj_to_numeric(df, ['user_score', 'user_pos', 'user_mixed', 'user_neg'])

df.loc[df['meta_overview'].isnull(), 'meta_overview'] = 'No score yet'
df.loc[df['user_overview'].isnull(), 'user_overview'] = 'No user score yet'

df.info()

df.describe().loc[['min', 'max'], ['meta_score', 'user_score']]

df['n_user_score'] = df['user_score'] * 10

df['release_date'] = df['release_date'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%b %d, %Y')))

df.to_csv('xboxone_games_cleaned.csv', index=False, encoding = 'utf-8')

"""#### Switch"""

df = pd.read_csv('switch_games.csv', lineterminator='\n')

df.info()

obj_to_numeric(df, ['meta_pos', 'meta_mixed', 'meta_neg', 'user_score', 'user_pos', 'user_mixed', 'user_neg'])

df.describe().loc[['min', 'max'], ['meta_score', 'user_score']]

df['n_user_score'] = df['user_score'] * 10

df['release_date'] = df['release_date'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%b %d, %Y')))

df.to_csv('switch_games_cleaned.csv', index=False, encoding = 'utf-8')

"""#### Merge

Let's merge the 3 .csv files into a single one called games.csv
"""

consoles = ['ps4', 'xboxone', 'switch']

tables = [pd.read_csv(f'{c}_games_cleaned.csv', lineterminator='\n') for c in consoles]

for t in tables: print(t.shape)

df = pd.concat(tables)

df.shape

df.to_csv('games.csv', index=False, encoding = 'utf-8')

"""## Reviews

#### Meta reviews

Load critics reviews of each platform
"""

meta_reviews = [pd.read_csv(f'{c}_meta_reviews.csv', lineterminator='\n') for c in consoles]

for t in meta_reviews: print(t.shape)

meta_reviews[0].head()

"""Same analysis as we did with games info to find missing values and incorrect data formats"""

for t in meta_reviews: print(t.info())

"""In the first dataframe there's a null score"""

meta_reviews[0].loc[meta_reviews[0]['score'].isnull()]

"""Let's delete its row"""

meta_reviews[0].drop(26547, inplace=True)

df = pd.concat(meta_reviews)

df['date'] = df['date'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%b %d, %Y')))

df.to_csv('meta_reviews.csv', index=False, encoding = 'utf-8')

"""#### User reviews"""

user_reviews = [pd.read_csv(f'{c}_user_reviews.csv', lineterminator='\n') for c in consoles]

for t in user_reviews: print(t.shape)

user_reviews[0].head()

for t in user_reviews: print(t.info())

user_reviews[0].loc[user_reviews[0]['text'].isnull()]

user_reviews[0].drop(15918, inplace=True)

df = pd.concat(user_reviews)

df['date'] = df['date'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%b %d, %Y')))

df.to_csv('user_reviews.csv', index=False, encoding = 'utf-8')

"""#### Merge"""

meta_df = pd.read_csv('meta_reviews.csv', lineterminator='\n')
user_df = pd.read_csv('user_reviews.csv', lineterminator='\n')

(meta_df.shape, user_df.shape)

meta_df.describe().loc[['min', 'max'], ['score']]

user_df.describe().loc[['min', 'max'], ['score']]

"""As in games info, critics and user scores are in different range."""

user_df['score'] = user_df['score'] * 10

df = pd.concat([meta_df, user_df]).reset_index(drop=True)

"""Let's read some reviews"""

def print_examples(df, qty=1):
    for i in range(qty):
        print(df.iloc[i]['text'])
        print('\n')

print_examples(df.loc[df['score'] > 85], 5)

print_examples(df.loc[df['score'] < 40], 5)


df['text'] = df.apply(lambda x: x.text.lower().replace(f'{(x["title"]).lower()}', ''), 1)



print_examples(df.loc[df['score'] < 40], 5)



df.iloc[132654]['text']

df.iloc[132648]['text']



def detect_lang(row):
    try:
        lang = detect(row.text)
    except:
        lang = "error"
    return lang

df['lang'] = df.apply(lambda x: detect_lang(x), 1)

df['lang'].value_counts()[:10]

"""We are only interested in english reviews"""

df = df.loc[df['lang'] == 'en']

df.to_csv('reviews.csv', index=False, encoding = 'utf-8')
