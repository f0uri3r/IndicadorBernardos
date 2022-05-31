
__author__  = "J-B. J. Fourier"
__version__ = "1.0"
__email__   = "f0uri3r@protonmail.es"

"""
Analysis of the Bernardos indicator correlation with BTC price.
"""


"""
IMPORTS
"""
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.signal import savgol_filter



import tweepy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import datetime as dt
import time
import urllib.request
import json


pd.set_option("display.max_rows", None, "display.max_columns", None)



def init_tweepy_api(consumer_key, consumer_secret):
    """Initializes an API object.

        Parameters
        ----------
        consumer_key : str
            Public key of the API user
        consumer_secret : str
            Secret key of the API user

        Raises
        ------
        api: tweepy.api.API
            The api object contains all the functionalities to use the Twitter API.
        """

    auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
    api = tweepy.API(auth)
    
    return api



def get_btc_tweets(api, usr_name, cum_btc_tweets_file):
    """Extracts the list of the BTC tweets of the selected account.

        Parameters
        ----------
        api : tweepy.api.API
            The api object contains all the functionalities to use the Twitter API.
        usr_name : str
            Name of the Twitter account
        cum_btc_tweets_file: str
            Name of the file that saves the BTC tweets

        Raises
        ------
        cum_btc_tweets: array
            Historic of BTC related tweets
        """

    keyword        = ["itcoin", "cripto"]
    all_tweets     = []
    all_btc_tweets = []
    all_btc_tweets_ids = []

    tweets = api.user_timeline(screen_name=usr_name, 
                           count=200,
                           include_rts = False,
                           tweet_mode = 'extended'
                           )

    all_tweets.extend(tweets)
    oldest_id = tweets[-1].id

    while True:
        tweets = api.user_timeline(screen_name=usr_name, 
                            count=200,
                            include_rts = False,
                            max_id = oldest_id - 1,
                            tweet_mode = 'extended'
                            )

        if len(tweets) == 0:
            break
        oldest_id = tweets[-1].id
        all_tweets.extend(tweets)

        print('{} tweets descargados'.format(len(all_tweets)))

    for Tweet in all_tweets:
        text = Tweet.full_text

        if any(x in text for x in keyword):
            all_btc_tweets.append(Tweet)
            all_btc_tweets_ids.append(Tweet.id)


    if os.path.isfile(cum_btc_tweets_file):

        cum_btc_tweets     = list(np.load(cum_btc_tweets_file, allow_pickle=True))
        cum_btc_tweets_ids = [tw.id for tw in cum_btc_tweets]
        new_tweets_ids     = set(all_btc_tweets_ids) - set(cum_btc_tweets_ids)

        if new_tweets_ids:
            for Tweet in all_btc_tweets:
                if Tweet.id in new_tweets_ids:
                    cum_btc_tweets.append(Tweet)

        np.save(cum_btc_tweets_file, cum_btc_tweets)

    else:

        cum_btc_tweets = all_btc_tweets
        np.save(cum_btc_tweets_file, all_btc_tweets)  

    return cum_btc_tweets



def dates_clustering(dates_btc_tweets, bandwidth):
    """Computes from BTC tweets dates a dendogram and finds clusters

        Parameters
        ----------
        dates_btc_tweets: datetime arrayo
            List of all dates of Bernardos BTC tweets
        bandwidth: scalar
            Width of the MeanShift search band (in days)

        Raises
        ------
        void        
        """
    
    n = len(dates_btc_tweets)
    D = np.zeros([n, n])
    dates_btc_tweets_str = [ date.strftime("%Y-%m-%d") for date in dates_btc_tweets ]

    # MeanShift
    days_btwn_dates = [ td.days for td in np.diff(dates_btc_tweets) ]
    days_btwn_dates.insert(0, 0)
    dates_ids = np.cumsum(days_btwn_dates)
    dates_ids = np.reshape(dates_ids, (-1, 1))

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(dates_ids)
    
    labels          = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique   = np.unique(labels)
    n_clusters_     = len(labels_unique)
    print("Number of estimated clusters : %d" % n_clusters_)
    df_meanshift = pd.DataFrame({"Dates": dates_btc_tweets_str,
                                 "Labels": labels,
                                 "Ncluster": np.zeros(len(labels)),
                                 "Color"   : np.zeros(len(labels))})
    
    for l in np.unique(df_meanshift["Labels"]):
        df_meanshift.iloc[ df_meanshift["Labels"] == l, 2 ] =  np.sum(df_meanshift["Labels"] == l)

    df_meanshift["Color"] =  ( (df_meanshift["Ncluster"] - df_meanshift["Ncluster"].min()) * (1/(df_meanshift["Ncluster"].max() - df_meanshift["Ncluster"].min()) * 255) ).astype(int)

    return df_meanshift


def kernel_density_estimation(dates_btc_tweets, kernel, bandwidth, gridsearch=False):
    """Returns a BTC tweets kernel density estimation model

        Parameters
        ----------
        dates_btc_tweets: datetime array
            List of all dates of BTC tweets
        kernel: string
            Kernel type
        bandwidth: scalar
            Width of the kernel (in days)
        gridsearch: bool
            Estimate the best model parameters.
        
        Raises
        ------
        kde_model: obj
            Kernel density estimation model        
        """

    dates_ordinal = np.array([x.toordinal() for x in dates_btc_tweets])

    if gridsearch:
        param_grid = {'kernel': ['gaussian', 'epanechnikov', 'exponential', 'linear'],
                    'bandwidth' : np.linspace(1, 60, 100)
                    }

        grid = GridSearchCV(
                estimator  = KernelDensity(),
                param_grid = param_grid,
                n_jobs     = -1,
                cv         = 10, 
                verbose    = 0
            )

        _ = grid.fit(X = dates_ordinal.reshape((-1,1)))

        print("----------------------------------------")
        print("Best hyperparameters (cv)")
        print("----------------------------------------")
        print(grid.best_params_, ":", grid.best_score_, grid.scoring)

        kde_model = grid.best_estimator_
        
    else:
        kde_model    = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        kde_model.fit(X=dates_ordinal.reshape(-1, 1))
        
    return (dates_ordinal, kde_model)



def candlestick_plot(dates_btc_tweets, usr_name, kde_info, df_clustering=None):
    """Shows BTC price with BTC tweets events

        Parameters
        ----------
        dates_btc_tweets: datetime array
            List of all dates of BTC tweets
        usr_name: string
            User name

        Raises
        ------
        void        
        """

    tvela  = '1w'
    id_btc = 'BTCUSDT'

    endTime         = dt.datetime.now()
    endTime_POSIX   = int(time.mktime(endTime.timetuple()) * 10**3)
    startTime       = dt.date(dates_btc_tweets[0].year, dates_btc_tweets[0].month, 1)
    startTime_POSIX = int(time.mktime(startTime.timetuple()) * 10**3)

    href   = 'https://api.binance.com/api/v3/klines?symbol=' + id_btc + '&interval=' + tvela + '&startTime=' + str(startTime_POSIX) + '&endTime=' + str(endTime_POSIX)
    page   = urllib.request.urlopen(href)
    data   = json.load(page)
    df_BTC = pd.DataFrame({"Date"   : [np.nan]*len(data),
                           "Open"   : [np.nan]*len(data),
                           "High"   : [np.nan]*len(data),
                           "Low"    : [np.nan]*len(data),
                           "Close"  : [np.nan]*len(data),
                           "Volume" : [np.nan]*len(data)
                          })

    for i in range(len(data)):
        df_BTC.iloc[i,0]   = dt.datetime.utcfromtimestamp(int(data[i][0]/10**3)) # Date
        df_BTC.iloc[i,1] = float(data[i][1]) # Open
        df_BTC.iloc[i,2] = float(data[i][2]) # High
        df_BTC.iloc[i,3] = float(data[i][2]) # Low
        df_BTC.iloc[i,4] = float(data[i][4]) # Close
        df_BTC.iloc[i,5] = float(data[i][5]) # Volume

    df_BTC["log_Open"]  = np.log(df_BTC["Open"])
    df_BTC["log_High"]  = np.log(df_BTC["High"])
    df_BTC["log_Low"]   = np.log(df_BTC["Low"])
    df_BTC["log_Close"] = np.log(df_BTC["Close"])
    BTC_smooth = savgol_filter(df_BTC["Close"], 14, 3)
    

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Candlestick(x     = df_BTC["Date"],
                                 open  = (df_BTC["Open"]),
                                 high  = (df_BTC["High"]),
                                 low   = (df_BTC["Low"]),
                                 close = (df_BTC["Close"]),
                                 name = 'BTC week candles'),
                    secondary_y = False)

    fig.add_trace(go.Scatter(mode = 'lines',
                             x = df_BTC["Date"],
                             y = BTC_smooth, 
                             line = {'color':'green', 'width':1},
                             name = 'Smoothed BTC price'
                            ),
                    secondary_y = False
                    
                  )

    
    dates_ordinal = np.array([x.toordinal() for x in df_BTC["Date"]])
    _, kde_model = kde_info

    density_pred = np.exp(kde_model.score_samples(dates_ordinal.reshape((-1,1))))
    

    fig.add_trace(go.Scatter(mode = 'lines',
                             x = df_BTC["Date"],
                             y = density_pred, 
                             line = {'color':'black', 'width':2},
                             name = 'Indicator KDE function'
                            ),
                    secondary_y = True
                  )
    
    fig.update_layout(title = '@' + usr_name + " indicator vs. BTC price")
    fig.update_yaxes(title_text="KDE", secondary_y=True)
    fig.update_yaxes(title_text="USD / BTC", secondary_y=False)
    
    print('Adding tweets to BTC price chart...')
    for date in tqdm(dates_btc_tweets):
        
        date_str = date.strftime("%Y-%m-%d")
        c = np.unique( df_clustering[ df_clustering["Dates"] == date_str]["Color"] )[0]

        fig.add_shape(type = "line", 
                      x0   = date_str,
                      x1   = date_str,
                      y0   = 0,
                      y1   = 1,
                      xref = "x",
                      yref = "paper",
                      line=dict(
                                color="rgb(" + str(c) +",100,0.7)",
                                width=1,
                                dash="dot",
                                )
                      )
    
    
    fig.show()


    f, ax = plt.subplots()
    ax.plot(df_BTC["Date"], density_pred, color="black")
    ax.set_xlabel('Date')
    ax.set_ylabel(r'Indicator $B(t)$', color="black")
    ax.grid(alpha=0.3)
    ax.set_title('Indicator vs. BTC price trends')

    ax2 = ax.twinx()
    ax2.plot(df_BTC["Date"], np.insert(np.diff(BTC_smooth), 0, 0, axis=0), color="C1", marker="o")
    ax2.set_ylabel(r'BTC price 1st derivative $P\'(t)$', color="C1")
    ax2.tick_params(axis='y', colors="C1")





""" EXECUTION """



# Users: GonBernardos, PeterSchiff
usr_name        = "GonBernardos" 
btc_tweets_file = "all_btc_tweets_" + usr_name + ".npy"

# Use 
get_tweets = False
# Your API keys (Not necessary if you don't want to update tweets and you have a .npy file)
consumer_key    = "2nWFzwNQCvj3ndnQLlYnTpFBQ"
consumer_secret = "N3xOp0I4FaiBiYkcuajlFM3FrcGTUizjSUUpCZLfQAacxJZXpT"

if get_tweets:
    api = init_tweepy_api(consumer_key, consumer_secret)
    all_btc_tweets = get_btc_tweets(api, usr_name, btc_tweets_file) 
else:
    all_btc_tweets = np.load(btc_tweets_file, allow_pickle=True)

# BTC tweets dataframe preprocessing
df_btc_tweets = pd.DataFrame({"Tweet": len(all_btc_tweets)*[None],
                                "Date": len(all_btc_tweets)*[None],
                                "Id": len(all_btc_tweets)*[None],
                                "Is_reply": len(all_btc_tweets)*[False]
                                })
for i in range(len(all_btc_tweets)):

    tw = all_btc_tweets[i]

    if tw.in_reply_to_status_id is not None:
        isreply = True
    else:
        isreply = False

    df_btc_tweets.iloc[i,:] = [tw.full_text, tw.created_at, tw.id, isreply]

if usr_name == "GonBernardos":
    df_old_btc_tweets         = pd.read_excel("old_btc_tweets.xlsx")
    df_old_btc_tweets["Date"] = pd.to_datetime( df_old_btc_tweets["Date"] )
    df_btc_tweets = pd.concat([df_old_btc_tweets, df_btc_tweets])

df_btc_tweets = df_btc_tweets.sort_values(by=["Date"])
df_btc_tweets = df_btc_tweets[df_btc_tweets["Is_reply"] == False]
dates_btc_tweets = df_btc_tweets["Date"].to_list()


# Discrete clustering of the dates
bandwidth     = 3 # days
df_clustering = dates_clustering(dates_btc_tweets, bandwidth)


# Indicator: kernel density estimation
bandwidth  = 14 # days
kernel     = 'gaussian' 
gridsearch = False
kde_model  = kernel_density_estimation(dates_btc_tweets, kernel, bandwidth, gridsearch)

# Results
candlestick_plot(dates_btc_tweets, usr_name, kde_model, df_clustering)

plt.show()