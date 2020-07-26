import pickle
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Test Model')
parser.add_argument('--test_data',
                       type=str,
                       help='Path to train csv file')
args = parser.parse_args()

test_path = args.test_data
print('loading model')

source_dict = pickle.load(open("source_dict.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

print('loading data')
df_test = pd.read_csv(test_path, sep='\t')



print('evaluating')
X_test = df_test['text'].fillna('').to_list()
X_test = tfidf.transform(X_test)
y_pred = np.array(model.predict(X_test))
source_text = df_test['source'].to_list()

y_pred2 = [source_dict.get(i, 0.5) for i in source_text]
pred = []
for i,j in zip(y_pred, y_pred2):
    pred.append(round((i+j*1)/2))
    
df_test['label'] = pred
df_test['label'] = df_test['label'].astype(int) 
df_test[['title', 'label']].to_csv('output.csv', sep='\t')

    
