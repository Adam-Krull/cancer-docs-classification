#imports
import argparse
import json
import pandas as pd
import pickle
import regex as re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

#parse filename argument from command line
parser = argparse.ArgumentParser(description='Return predicted class for documents.')
parser.add_argument('-f', '--file', help='CSV with text data.')
args = vars(parser.parse_args())
filename = args['file']

#vectorizer object
with open('vec.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

#columns to keep after vectorizing
with open('feature_list.json', 'r') as f:
    col_bools = json.load(f)     

#xgboost model
with open('xgb.pickle', 'rb') as f:
    xgb = pickle.load(f)

#label encoder for predictions
with open('enc.pickle', 'rb') as f:
    encoder = pickle.load(f)

#define global variables
unpacked_stopwords = stopwords.words('english')
ps = PorterStemmer()    

#function to clean data
def clean_text(df):
    col = df.columns[0]
    regexp = r"[^a-zA-Z\s']"
    df = df[col].str.replace(regexp, "", regex=True)
    df = df.str.lower()

    return df

#function to remove stopwords / stem words
def remove_stopwords(article, stopword_list=unpacked_stopwords, stemmer=ps):

    tok_article = word_tokenize(article)
    approved_words = []

    for word in tok_article:
        if word in stopword_list:
            continue
        else:
            stem = stemmer.stem(word)
            approved_words.append(stem)

    return " ".join(approved_words)

#function to create features
def create_features(df, vec=vectorizer, cols=col_bools):
    
    df_vec = vec.transform(df)
    df_vec = pd.DataFrame(df_vec.toarray(), columns=vec.vocabulary_.keys())
    
    return df_vec.loc[:, cols]

#function to make and return predictions
def make_preds(df, model=xgb, enc=encoder):

    preds = model.predict(df)
    return enc.inverse_transform(preds)

#run it
if __name__ == '__main__':
    #read in data
    data = pd.read_csv(filename, encoding='MacRoman', index_col=0)
    #clean data
    clean_data = clean_text(data)
    #prep
    prep_data = clean_data.apply(remove_stopwords)
    #create features
    features = create_features(prep_data)
    #get preds
    preds = make_preds(features)
    #print em
    print(preds)