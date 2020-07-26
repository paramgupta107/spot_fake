from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.svm import LinearSVC
import pickle
import argparse

parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--train_data',
                       type=str,
                       help='Path to train csv file')
args = parser.parse_args()

train_path = args.train_data



print('loading data')
df = pd.read_csv(train_path, sep='\t')
X_train = df['text'].fillna('').to_list()


y_train = df['label'].to_list()


print('feature vector')
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
tfidf.fit(X_train)
features = tfidf.transform(X_train).toarray()

print(features.shape)


print('model train')


model = LinearSVC()

model.fit(features, y_train)
def parse_url(x):
    source = x['source']
    source = source.replace('https://','')
    source = source.replace('http://','')
    source = source.replace('www1.','')
    source = source.replace('www2.','')
    source = source.replace('www3.','')
    source = source.replace('www.','')
    return source

df['source'] = df.apply(parse_url, axis=1)

source_dict = df.groupby('source').mean()['label'].to_dict()

print('Saving model')
pickle.dump(source_dict, open("source_dict.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
