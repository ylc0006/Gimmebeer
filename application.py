# from __future__ import division
# from flask import Flask, render_template, request
# import os
# import io
# import csv
# import sys
# import pandas as pd
# from pathlib import Path
# import nltk, re
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# import string
# from nltk.stem.porter import PorterStemmer
# from nltk.probability import FreqDist
# import matplotlib.pyplot as plt
# from PIL import Image
# from wordcloud import WordCloud
# import numpy as np
# import urllib
# import requests
# import warnings
# from sklearn.preprocessing import MinMaxScaler
from __future__ import division
from flask import Flask, render_template, request
import io
import csv
import json
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from PIL import Image
from wordcloud import WordCloud
import numpy as np
import requests
from math import sqrt
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

def get_similarity(user, df):

    # given a userID, calculate user-user correaltion matrix
    # if userID is a new ID, add userID-beerID-ratings to df before running get_similarity()
    # return dataframe[userID, beerID, ratings, similarity] where similarity>0

    # rating_pivot = df.pivot(index='beerID', columns='userID', values='ratings')
    rating_pivot = pd.read_csv('data/pivot_ratings.csv', index_col=0)

    #user = 'beerguy101'
    bone_rating = rating_pivot[user]

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    similar_to_bones = rating_pivot.corrwith(bone_rating)
    corr_bone = pd.DataFrame(similar_to_bones, columns=['similarity'])
    corr_bone1 = corr_bone.dropna()
    corr_bone2 = corr_bone1.loc[corr_bone1['similarity'] > 0]
    # corr_bone = corr_bone.loc[corr_bone1['similarity']==1]
    corr_bone3 = corr_bone2.sort_values('similarity', ascending=False)
    corr_bone3['userID'] = corr_bone3.index

    # combine df1[userID, similarity] with df2[userID, beeriD, ratings] to get beerIDs ratied by users in df1
    df_simi_temp = df.join(corr_bone3, on='userID', rsuffix='_simi')
    df_simi = df_simi_temp.drop('userID_simi', axis=1)
    df_simi = df_simi.dropna()

    return df_simi

def get_recommend(beer_rate_by_user, simi_temp, beer_rate_count):
    warnings.filterwarnings('ignore', category=FutureWarning)


    # exclude beers in df_similarity(return of get_similarity())
    score_temp1 = simi_temp[~simi_temp['beerID'].isin(beer_rate_by_user)]

    # calculate sum(similarity*ratings)/sum(similarity) for each beerID
    score_temp2 = score_temp1['similarity'] * score_temp1['ratings']
    score_temp3 = pd.Series.to_frame(score_temp2)
    score_temp3.columns = ['simi_score']
    score_temp = score_temp1.join(score_temp3, rsuffix='_multiply')
    scores = score_temp.groupby('beerID')['simi_score'].sum()
    beer_score_sum = pd.Series.to_frame(scores)
    simi_sum = score_temp.groupby('beerID')['similarity'].sum()
    beer_simi_sum = pd.Series.to_frame(simi_sum)

    beer_score = beer_score_sum.join(beer_simi_sum, rsuffix='_simi')
    beer_score['score'] = beer_score['simi_score'] / beer_score['similarity']
    beer_score = beer_score.sort_values(by='score', ascending=0)
    beer_score['beerID'] = beer_score.index.astype(int)

    df_score = beer_score.join(beer_rate_count, on='beerID', rsuffix='_count')
    df_score = df_score.drop('beerID_count', axis=1)

    # filter rating counts>300, return df[beerID, score, rating_count]
    df_recommend_temp = df_score[['beerID', 'score', 'rating_count']]
    filter_recommend = df_recommend_temp.loc[df_recommend_temp['rating_count'] >= 200]
    df_recommend = filter_recommend.sort_values(by=['score','rating_count'], ascending=[False,False])
    # df_recommend_score = df_recommend[['beerID', 'score']]

    return df_recommend

@app.route('/')
def index():
    return render_template('beer.html')

@app.route('/userid', methods=['POST'])
def recommend_by_userid():

    infilename = 'data/beeradvocate_tableformat.tsv'
    mappingfile = 'data/mapping.csv'
    chunksize = 10**5

    #Zixin's code
    #user = 'beerguy101'
    user = str(request.form['userid'])
    filename = 'data/clean_ratings.csv'
    df_rating = pd.read_csv(filename)
    df_rating.columns = ['userID', 'beerID', 'ratings']

    df_mapping = pd.read_csv(mappingfile,encoding='latin-1')

    # user = 'xxxxx123'
    df = df_rating

    user_list = df_rating['userID'].unique()

    if user in user_list:

        simi_temp = get_similarity(user, df)

        # find beers rated by this user(to exclude these beers)
        beer_user = df.loc[df['userID'] == user]
        beer_rate_by_user = beer_user['beerID'].tolist()
        # count number of ratings for each beerID, add to beer_score
        beer_rate_count1 = df.groupby('beerID')['ratings'].count()
        beer_rate_count = pd.Series.to_frame(beer_rate_count1)
        beer_rate_count.columns = ['rating_count']
        beer_rate_count['beerID'] = beer_rate_count.index

        recommend = get_recommend(beer_rate_by_user, simi_temp, beer_rate_count)
        recommend_top10 = recommend.head(10)
        recommend_beer = recommend_top10.rename(columns={'beerID':'beerid'})

        #load original data
        data_raw = pd.read_csv(infilename,sep="\t",index_col=False)
        data_raw.columns
        beer_text = data_raw[['beerid','text','style']]
        top10 = pd.merge(recommend_beer, beer_text, on='beerid')
        #pick 10 users comments per beer for text analysis
        top10 = top10.groupby('beerid').head(100).reset_index(drop=True)
        beer_name = data_raw[['beerid','name','style']]
        beer_top10 = pd.merge(recommend_beer,beer_name,on='beerid')
        top10_df = beer_top10[['name','score','style','beerid','rating_count']].drop_duplicates()
        top10_df = top10_df.join(df_mapping.set_index('style'), on='style')

        #scale rating count to (0.2, 1)
        scaler = MinMaxScaler(feature_range=(0.2, 1))
        top10_df[['rating_count']] =  scaler.fit_transform(top10_df[['rating_count']])

        top10_rate = []
        for index, beer in top10_df.iterrows():
            top10_rate.append([beer['name'],int(beer['score']*20),beer['style_new'],beer['beerid'],beer['rating_count']])
        for i in top10_rate:
            rank_in_size = 100-10*(top10_rate.index(i))
            i.append(rank_in_size)

        print(top10_rate)
        # print(time.time()-st_time)

    else:
        print('new user, need init')



    ################## word cloud #####################

    #extract text from beer reveiw
    raw =[]
    for index, beer in top10.iterrows():
        raw.append(beer['text'])

    # join into one big string
    raw_inline = ' '.join(raw)

    # tokenize
    tokens = word_tokenize(raw_inline)

    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    stop_words = stopwords.words('english') # stop words from nltk library

    add_sw = ['reviewtext', 'beer', 'one', 'would','bit','overall','im','way','oz',
          'two','id','see','come','say','head','nice','good','like','little','well',
          'really','much', 'great','notes','pretty','quite','drink','slight','almost',
          'first','still','decent','though','quickly','finger','beers','feel','even',
          'lots','around','bad','definitely','back','small','lot','best','try','nothing',
          'dont','enough','amount','end','maybe','high','doesnt','something','drinking',
          'top','another','theres','right','seems','interesting','make','sure','nicely',
          'thanks','going','left','tap','rather','side','somewhat','average','low',
          'isnt','got','pleasant','along','faint','expected','goes','go','yet','thats',
          'probably','cant','love','kind','didnt','many','enjoyable','actually','note',
          'find','know','mostly','thing','ever','without','less','ml','session',
          'day','never','certainly','away','however','want','enjoy','give','real','review',
          'highly','easily','ill','coming','wasnt','else','ok']

    for i in add_sw:
        stop_words.append(i) # add more stop words


    words = [w for w in words if not w in stop_words]

    # stemming of words
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]

    # frequency distribuition of words
    fdist = FreqDist(stemmed)

    # take top words
    top = fdist.most_common(100) # free to adjust the number

    dict_final_word_count = {}
    for i in range(len(top)):
        dict_final_word_count[top[i][0]] = top[i][1]

    # beer shape word cloud
    mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/f/e/f/c/11949907341218632906beer.svg.med.png', stream=True).raw))
    wc = WordCloud(width= 512, height =512, max_font_size=25, background_color="white", max_words=1000, mask=mask, contour_width=1, contour_color='firebrick')

    # Generate a wordcloud
    wc.generate_from_frequencies(dict_final_word_count)

    # store to file
    #wc.to_file("Vis/static/beer.png")
    wc.to_file("static/beer.png")


    return render_template("recommend_by_attr.html", data=top10_rate)

@app.route('/attribute', methods=['POST'])
def recommend_by_attribute():

    ### calucate user input attribute weight
    appearance = float(request.form['appearance'])
    aroma = float(request.form['aroma'])
    palate = float(request.form['palate'])
    taste = float(request.form['taste'])
    style = str(request.form['style'])

     #appearance = 4
     #aroma = 2
     #palate = 4
     #taste = 5
     #style = 'All'

    sum = float(appearance + aroma + palate + taste)

    appearnce_w = appearance / sum
    aroma_w = aroma / sum
    palate_w = palate / sum
    taste_w = taste / sum

    ### calcluate sum of squred difference for user input vs. dataset
    infilename = 'data/UserWeight.tsv'

    data = []
    with io.open(infilename, encoding='utf-8') as infile:
        rd = csv.reader(infile, delimiter='\t')
        for row in rd:
            data.append(row)

    data.pop(0)

    result = []
    for item in data:
        appearance = float(item[1])
        aroma = float(item[2])
        palate = float(item[3])
        taste = float(item[4])
        squared_difference = (appearance - appearnce_w)**2 + (aroma - aroma_w)**2 + (palate - palate_w)**2 + (taste - taste_w)**2
        result.append([item[0], squared_difference, (1-squared_difference)*100]) # Add similar user score

    def getkey(item):
        return item[1]

    sorted_result = sorted(result, key=getkey) # sort by sum of squared differenice, asecending

    users = sorted_result[:200] ##### get the top 200 similar users
    for i in range(len(users)): # add similar user score
        users[i].append(20-0.2*i)


    ### extract beers from similar users
    beers = []
    for user in users:
        infilename = 'data/UserScoreTextStyle/' + user[0] + '.tsv'
        with io.open(infilename, encoding='utf-8') as infile:
            rd = csv.reader(infile, delimiter='\t')
            next(rd, None)
            for row in rd:
                beers.append(row)

    ### add total score for the recommend beer ("UserScore" * "BeerRating")
    users_dict = {}
    for user in users:
        users_dict[user[0]] = user[3]

    recommend_beers =[]
    for beer in beers:
        if len(beer) == 13:
            recommend_beers.append([beer[0],beer[1],beer[2],beer[3],beer[4],beer[5],beer[6],beer[7],beer[8],beer[9],beer[10],beer[11],beer[12],round((users_dict[beer[0]] * float(beer[1]))**3/10000)])

    ### filter beers by selected stylename
    beers_style =[]
    if style == "All":
        for beer in recommend_beers:
            beers_style.append([beer[6],beer[13],beer[11],beer[7]])
    else:
        for beer in recommend_beers:
            if beer[-3] == style:
                beers_style.append([beer[6],beer[13],beer[11],beer[7]])


    #print(beers_style)
    # sort by overall rating, descending
    def getkey(item):
        return item[1]

    sorted_beer = sorted(beers_style, key=getkey, reverse = True)
    top10_rate = sorted_beer[:10]
    for i in top10_rate:
        rank_in_size = 100-10*(top10_rate.index(i))
        i.append(rank_in_size)

    #load beer count file
    beer_count_file = 'data/beer_counts.csv'
    df_beercount = pd.read_csv(beer_count_file)
    df_beercount.columns = ['beerid', 'reviewcount']
    df_beercount['beerid']=df_beercount['beerid'].apply(int)

    #conver top10_rate to df
    df_top10_rate = pd.DataFrame(np.array(top10_rate), columns = ['beername','rating','style','beerid','sizescore'])
    df_top10_rate['beerid'] = df_top10_rate['beerid'].apply(int)

    #join table to find count of review for each beer
    df_top10_rate = df_top10_rate.join(df_beercount.set_index('beerid'), on='beerid')

    #rearrange column order and convert it back to list for visiulization
    df_top10_rate = df_top10_rate[['beername','rating','style','beerid','reviewcount','sizescore']]

    #scale rating count to (0.2, 1)
    scaler = MinMaxScaler(feature_range=(0.2, 1))
    df_top10_rate[['reviewcount']] =  scaler.fit_transform(df_top10_rate[['reviewcount']])

    top10_rate = df_top10_rate.values.tolist()

    ################## word cloud #####################
    #extract text from beer reveiw
    raw =[]
    for beer in top10_rate:
        raw.append(beer[0])

    # join into one big string
    raw_inline = ' '.join(raw)

    # tokenize
    tokens = word_tokenize(raw_inline)

    # convert to lower case
    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    stop_words = stopwords.words('english') # stop words from nltk library

    add_sw = ['reviewtext', 'beer', 'one', 'would','bit','overall','im','way','oz',
          'two','id','see','come','say','head','nice','good','like','little','well',
          'really','much', 'great','notes','pretty','quite','drink','slight','almost',
          'first','still','decent','though','quickly','finger','beers','feel','even',
          'lots','around','bad','definitely','back','small','lot','best','try','nothing',
          'dont','enough','amount','end','maybe','high','doesnt','something','drinking',
          'top','another','theres','right','seems','interesting','make','sure','nicely',
          'thanks','going','left','tap','rather','side','somewhat','average','low',
          'isnt','got','pleasant','along','faint','expected','goes','go','yet','thats',
          'probably','cant','love','kind','didnt','many','enjoyable','actually','note',
          'find','know','mostly','thing','ever','without','less','ml','session',
          'day','never','certainly','away','however','want','enjoy','give','real','review',
          'highly','easily','ill','coming','wasnt','else','ok']

    for i in add_sw:
        stop_words.append(i) # add more stop words

    words = [w for w in words if not w in stop_words]

    # stemming of words
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]

    # frequency distribuition of words
    fdist = FreqDist(stemmed)

    # take top words
    top = fdist.most_common(100) # free to adjust the number

    dict_final_word_count = {}
    for i in range(len(top)):
        dict_final_word_count[top[i][0]] = top[i][1]

    # beer shape word cloud
    mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/f/e/f/c/11949907341218632906beer.svg.med.png', stream=True).raw))
    wc = WordCloud(width= 512, height =512, max_font_size=25, background_color="white", max_words=1000, mask=mask, contour_width=1, contour_color='firebrick')

    # Generate a wordcloud
    wc.generate_from_frequencies(dict_final_word_count)

    # store to file
    #wc.to_file("Vis/static/beer.png")
    wc.to_file("static/beer.png")


    return render_template("recommend_by_attr.html", data=top10_rate)

#     return '''
# <html>
#     <head>
#         <title>Beer Recommend</title>
#     </head>
#     <body>
#         <h1>Beer Recommend</h1>
#         <div><p>No. 1: %s <p><div>
#         <div><p>No. 2: %s <p><div>
#         <div><p>No. 3: %s <p><div>
#         <div><p>No. 4: %s <p><div>
#         <div><p>No. 5: %s <p><div>
#         <div><p>No. 6: %s <p><div>
#         <div><p>No. 7: %s <p><div>
#         <div><p>No. 8: %s <p><div>
#         <div><p>No. 9: %s <p><div>
#         <div><p>No. 10: %s <p><div>
#         <a href="/">Try Again</a>
#    </body>
# </html>''' % (top10[0][6], top10[1][6], top10[2][6], top10[3][6], top10[4][6], top10[5][6], top10[6][6], top10[7][6], top10[8][6], top10[9][6] )



if __name__ == '__main__':
    app.run()#(host = '0.0.0.0', port = 3000)
    # app.run()
