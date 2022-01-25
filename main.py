import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
import time
import pandas as pd
import string

#for removing stop words and non english words
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

#for stemming and lemitize
from nltk.tokenize import sent_tokenize, word_tokenize

#nltk.download('wordnet')

import numpy as np

#for removing non english
from langdetect import detect



# credentials*********************************************************

cred = credentials.Certificate("serviceAccountKey.json")

def save_to_json(data_table, file_name):
    with open(file_name, "w", encoding='utf-8') as file:
        json.dump(data_table, file, indent=4, ensure_ascii=False)
        print("Done")

def saving_logs(list_of_news, comments_count, news_count):
    logs = []
    logs.append({'Date': time.time(),
                 'Total news': news_count - 1,
                 'Total comments': comments_count,
                 'Loaded news': list_of_news})

    with open(logs_file_name, "w", encoding='utf-8') as file:
        json.dump(logs, file, indent=4, ensure_ascii=False)
        print("Done")


if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    default_app = firebase_admin.initialize_app(cred)
db = firestore.client()


def load_news(table_name):
    news_count = 1
    comments_count = 0
    collection = db.collection(table_name)
    docs = collection.stream()
    loaded_news = []

    try:
        for doc in docs:
            loaded_news.append(doc.to_dict())
            print('news ' + str(news_count) + ' loaded')
            news_count += 1
            #if news_count >= 3:
            #    break
    except:
        print('Error in load news')
        pass

    data_table = []
    list_of_news = []

    try:
        for i in range(0,len(loaded_news)):
            award_table = []

            ID = loaded_news[i]['ID']
            Date = loaded_news[i]['Date']
            Name = loaded_news[i]['Name']
            Text = loaded_news[i]['Text']
            Score = loaded_news[i]['Score']
            Ratio = loaded_news[i]['Ratio']
            Comments_num = loaded_news[i]['Comments_num']
            Page_url = loaded_news[i]['Page_url']
            comment_data = loaded_news[i]['Comments_list']
            award_data = loaded_news[i]['Awards_list']

            for a in range(0, len(award_data)):
                name = award_data[a]["name"]
                id = award_data[a]["id"]
                description = award_data[a]["description"]
                coin_price = award_data[a]["coin_price"]
                count = award_data[a]["count"]

                award_table.append({'name': name,
                                    'description': description,
                                    'coin_price': coin_price,
                                    'count': count,
                                    'id' : id})

            for k in range(0,len(comment_data)):

                Is_submitter = comment_data[k]["Is_submitter"]
                Comment_text = comment_data[k]["Comment"]
                try:
                    Comment_Score = comment_data[k]["Score"]
                except:
                    Comment_Score = comment_data[k]["Comment_Score"]
                Sticked = comment_data[k]["Sticked"]
                Distinguished = comment_data[k]["Distinguished"]

                comments_count += 1

                data_table.append({ 'ID':ID,
                                    'Date': Date,
                                    'Name': Name,
                                    'Text': Text,
                                    'Score': Score,
                                    'Ratio': Ratio,
                                    'Comments_num': Comments_num,
                                    'Page_url': Page_url,
                                    'Comment Date': Date,
                                    'Comment text': Comment_text,
                                    'Is_submitter': Is_submitter,
                                    'Comment score': Comment_Score,
                                    'Comment sticked': Sticked,
                                    'Comment distinguished': Distinguished,
                                    'Awards information': award_table})


            print(Name)
            list_of_news.append(ID)

    except:
        print('error in coverting to json')
        pass
    return Name, list_of_news, comments_count, news_count, data_table


def panda_clear_data(merged_df,Cleared_data_file_name):
    # drop duplicates
    merged_df = merged_df.drop_duplicates(subset=['Comment text'])

    # Remove emojis
    merged_df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))

    # make lowercase
    merged_df['Comment text'] = merged_df['Comment text'].str.lower()
    merged_df['Name'] = merged_df['Name'].str.lower()

    # remove ascii and other special symbols
    merged_df['Comment text'] = merged_df['Comment text'].str.replace('\n', '')
    merged_df['Comment text'] = merged_df['Comment text'].str.replace('\t', '')
    merged_df['Comment text'] = merged_df['Comment text'].str.replace(' {2,}', '', regex=True)
    merged_df['Comment text'] = merged_df['Comment text'].str.strip()
    merged_df['Comment text'] = merged_df['Comment text'].str.replace('[{}]'.format(string.punctuation), '', regex=True)
    merged_df['Comment text'] = merged_df['Comment text'][
        ~merged_df['Comment text'].str.contains(r'[^\x00-\x7F]', na=False)]

    merged_df['Name'] = merged_df['Name'].str.replace('\n', '')
    merged_df['Name'] = merged_df['Name'].str.replace('\t', '')
    merged_df['Name'] = merged_df['Name'].str.replace(' {2,}', '', regex=True)
    merged_df['Name'] = merged_df['Name'].str.strip()
    merged_df['Name'] = merged_df['Name'].str.replace('[{}]'.format(string.punctuation), '', regex=True)
    merged_df['Name'] = merged_df['Name'][~merged_df['Name'].str.contains(r'[^\x00-\x7F]', na=False)]

    # Filter only english
    def detect_en(text):
        try:
            return detect(text) == 'en'
        except:
            return False

    merged_df = merged_df[merged_df['Comment text'].apply(detect_en)]
    merged_df = merged_df[merged_df['Name'].apply(detect_en)]

    # remove stop words
    stop_words = stopwords.words('english')
    merged_df['Comment text'] = merged_df['Comment text'].fillna("")
    merged_df['Comment text'] = merged_df['Comment text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    merged_df['Name'] = merged_df['Name'].fillna("")
    merged_df['Name'] = merged_df['Name'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # Remove URLs
    merged_df['Comment text'] = merged_df['Comment text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '',
                                                                                                      regex=True)
    merged_df = merged_df.drop('Page_url', 1)

    # drop rows with no comments
    merged_df['Comment text'].replace('', np.nan, inplace=True)
    merged_df.dropna(subset=['Comment text'], inplace=True)

    # lemmitization
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(text):
        return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

    merged_df['Comment text'] = merged_df['Comment text'].apply(lemmatize_text)
    merged_df['Name'] = merged_df['Name'].apply(lemmatize_text)


    merged_df.to_csv(Cleared_data_file_name, index=False)


#********************************************

if __name__ == "__main__":
    # Parameters**********************************************************
    table_name = "Reddit_news_mk2"
    #file_name = "Reddit_news_mk2.json"
    logs_file_name = "logs.json"
    Cleared_data_file_name = 'Cleared_data_main.csv'

    #merged_df = pd.read_json('Reddit_news_mk2.json')
    #merged_df.to_csv(Cleared_data_file_name, index=False)

    (Name, list_of_news, comments_count, news_count, data_table) = load_news(table_name)

    saving_logs(list_of_news, comments_count, news_count)

    #save_to_json(data_table)

    merged_df = pd.json_normalize(data_table)
    panda_clear_data(merged_df, Cleared_data_file_name)

    print(list_of_news)