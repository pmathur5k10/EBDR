import sys
import jsonpickle
import os
import tweepy
import csv
import json
import re
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS
from preprocess_tweets import tweet_clean
import joblib
from flask import Flask
from flask_restful import Api, Resource, reqparse
import jsonpickle
import os
import tweepy
import csv
import json
import re
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS
import time
import threading
import pandas as pd

#app = Flask(__name__)
#api_1 = Api(app)


def tokenize_glove_func(text):
  
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", '' )
    text = re_sub(r"/", '')
    text = re_sub(r"@\w+", '')
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), '')
    text = re_sub(r"{}{}p+".format(eyes, nose), '')
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), '' )
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), '')
    text = re_sub(r"<3", '')
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", '' )
    text = re_sub(r"#\S+", '')
    text = re_sub(r"([!?.]){2,}",'')
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b",'')
    text = re_sub(r"([A-Z]){2,}", '')
    text = ''.join([c for c in text if c not in punctuation])
    words = text.split()

    return len(words)

def preprocess(text):


    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "")
    text = re_sub(r"/","")
    text = re_sub(r"@\w+", "")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "")
    text = re_sub(r"{}{}p+".format(eyes, nose), "")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "")
    text = re_sub(r"<3","")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "")
    text = " ".join(word.strip() for word in re.split('#|_', text))
    text = re_sub(r"([!?.]){2,}", r"")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"")
    

    text=text.lower()
    text=tweet_clean(text)
    return text

consumer_key='BGgRVICAeeB9qlVNnq1o5SQmH'
consumer_secret='vOBKrQEzZYwpaAOzWRsMGKhYmc56jZtLrXH6Wh03nOtGCVM4KA'

access_token='1039541660382707713-k4GaaNvvXRjV7iPyifv3TqQ90cjCzm'
access_token_secret='lgq7TWGAGr2XkWk16cGZ5Nv2fV83CC2OsBm7Yq9XL9kR9'
# Consumer and access credentials from twitter developers for access grant

auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth, parser=tweepy.parsers.JSONParser())

FLAGS = re.MULTILINE | re.DOTALL

#searchQuery = input("enter the query to be analysed....")  # this is what we're searching for

# We'll store the tweets in a text file.

maxTweets = 33 # Some arbitrary large number
tweetsPerQry = 33  # this is the max the API permits
fName = 'RBC.txt' 
fNamecsv= 'RBC.csv'


# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = None

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = 0
tweet_dict={}
tweet_dict['tweets']=[]
tweet_list=[]

check=[]

clf = joblib.load('clf.joblib')
clf2 = joblib.load('svc.joblib')

T = time.time()
def refresh():
    global T
    os.system("python3 twitter_create_dataset.py")
    T=time.time()
    while(1):
        C = time.time()
        if((C-T)>3600):
            os.system("python3 twitter_create_dataset.py")
            T=time.time()
Th = threading.Thread(target=refresh)
Th.start()
'''
class User(Resource):
    def get(self):
        """
        toret={}
        saved=0
        f = open("query_list.txt","r")
        while(1):
            global tokenize_glove_func
            searchQuery = f.readline()
            if(not searchQuery):
                break
            searchQuery = str(searchQuery)[:-1]
            print(searchQuery)
            tweetCount = 0
            while tweetCount < maxTweets:
                try:
                    if (max_id <= 0):
                        if (not sinceId):
                            new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                        else:
                            new_tweets = api.search(q=searchQuery, count=tweetsPerQry, since_id=sinceId)
                    else:
                        if (not sinceId):
                            new_tweets = api.search(q=searchQuery, count=tweetsPerQry, max_id=str(max_id - 1))
                        else:
                            new_tweets = api.search(q=searchQuery, count=tweetsPerQry, max_id=str(max_id - 1), since_id=sinceId)
                    if not new_tweets:
                        print("No more tweets found")
                        break

                    # print(new_tweets)    
                    for tweet in range(len(new_tweets["statuses"])):

                        # print(new_tweets["statuses"][tweet]["text"])

                        if new_tweets["statuses"][tweet]["text"] in set(check) or tokenize_glove_func(new_tweets["statuses"][tweet]["text"])<3:
                            continue


                        temp=[]
                        temp.append(new_tweets["statuses"][tweet]["id"])
                        temp.append(new_tweets["statuses"][tweet]["text"])
                        tweet_list.append(temp)
                        check.append(new_tweets["statuses"][tweet]["text"])

                        temp=new_tweets["statuses"][tweet]["text"]
                        temp=preprocess(temp)
                        pred=clf.predict([temp])
                        if(pred[0]==1):
                            print(pred,temp)
                            toret[saved]=temp
                            tweet_dict['tweets'].append({


                                'text' : new_tweets["statuses"][tweet]["text"],
                                'id' : new_tweets["statuses"][tweet]["id"],
                                'retweet_count': new_tweets["statuses"][tweet]["retweet_count"],
                                'retweeted' : new_tweets["statuses"][tweet]["retweeted"],
                                'source_of_posting' : new_tweets["statuses"][tweet]["source"],
                                'language' : new_tweets["statuses"][tweet]["lang"],
                                'place_of_posting' : new_tweets["statuses"][tweet]["place"],
                                'geo' : new_tweets["statuses"][tweet]["geo"],
                                'user_location' : new_tweets["statuses"][tweet]["user"]["location"],
                                'user_friends_count' : new_tweets["statuses"][tweet]["user"]["friends_count"],
                                'statuses_count' : new_tweets["statuses"][tweet]["user"]["statuses_count"],
                                'user_followers_count' : new_tweets["statuses"][tweet]["user"]["followers_count"],
                                'user_favourites_count' : new_tweets["statuses"][tweet]["user"]["favourites_count"],
                                'url_count' : len(new_tweets["statuses"][tweet]["entities"]["urls"]),
                                'url' : new_tweets["statuses"][tweet]["entities"]["urls"],
                                'hashtag_count' : len(new_tweets["statuses"][tweet]["entities"]["hashtags"]),
                                'hashtag' : new_tweets["statuses"][tweet]["entities"]["hashtags"],
                                'user_mention_count' : len(new_tweets["statuses"][tweet]["entities"]["user_mentions"]),
                                'user_mention' : new_tweets["statuses"][tweet]["entities"]["user_mentions"],
                                'special_symbols_count' : len(new_tweets["statuses"][tweet]["entities"]["symbols"]),
                                'special_symbols' : new_tweets["statuses"][tweet]["entities"]["symbols"],
                            }) 
                            saved+=1   




                        # print(tweet_dict["tweets"])
                        # json.dump(new_tweets["statuses"][0], f)
                        # q=input("wait")
                        # break
                    # break    



                        # f.write(jsonpickle.encode(tweet._json, unpicklable=False) +
                        #         
                        # print(tweet.keys())
                        
                        # tweet_dict.append(tweet)

                    tweetCount += len(new_tweets["statuses"])
                    print(tweetCount)    

                except tweepy.TweepError as e:
                    print("some error : " + str(e))
                    

        print ("Done")
        f.close()


        print(saved)
        print(len(tweet_list))
        with open(fName, 'a') as f:
            json.dump(tweet_dict['tweets'], f)

        with open(fNamecsv, 'a', encoding="utf-8") as f2:
            writer = csv.writer(f2, delimiter=',')
            for row in tweet_list:
                writer.writerow(row)
        """
        with open('RBC.csv',"r") as f:
            reader = csv.reader(f,delimiter = ",")
            data = list(reader)
            row_count = len(data)

        df = pd.read_csv('RBC.csv', skiprows = row_count - 10)
        print(df)
        return df.to_json(),200


api_1.add_resource(User, "/tweet/")

app.run(host='127.0.0.1')
'''
# print("Downloading max {0} tweets".format(maxTweets))
            
        
# tweet_dict_set=set(tweet_dict)
# tweet_dict_list=list(tweet_dict_set)
# print(tweet_dict_list)

# with open(fName, 'a', newline='') as myfile:
#     wr = csv.writer(myfile , delimiter=',')
#     for each_tweet in tweet_dict_list:
#         wr.writerow([each_tweet])
#     # wr.writerow(tweet_dict_list)