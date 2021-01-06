# Evaluation of various natural language processing models for fake news detection.


# Usage

Setup
```
git clone https://github.com/paramgupta107/spot_fake.git
cd spot_fake
pip install -r requirements.txt
```

Training 
```
python3 train.py --train_data <dataset path>
```

Testing 
```
python3 eval.py --test_data <dataset path>
```
Output will be saved as output.csv

# Metrics on 33% validation set

| Model | F1 Score  |
| ------------- | ------------- |
| TfidfVectorizer+Linear SVC+Source Confidence  | 0.88  |
| TfidfVectorizer+Linear SVC  | 0.87  |
| TfidfVectorizer+Dense  | 0.86  |
| Bert, USE, Infersent Concat + Dense  | 0.86  |
| Bi-LSTM + Glove  | 0.85  |
| USE + Dense  | 0.85  |
| Bi-LSTM  | 0.84  |
| Infersent + Dense  | 0.83  |
| Bert Embedding + Dense  | 0.83  |


[Detailed Metrics for these and various other tested models](/results.xlsx)

