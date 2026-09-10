import urllib.request
import os
import pandas as pd

urls = [
    "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full_corpus.csv",
    "https://raw.githubusercontent.com/sharmistha-b/Twitter-Sentiment-Analysis/main/Twitter_Data.csv",
    "https://raw.githubusercontent.com/pynecone-io/pynecone-examples/main/twitter_sentiment/twitter_sentiment.csv",
    "https://raw.githubusercontent.com/ankur3107/sentiment_analysis/master/financial_phrasebank.csv",
    "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/train_text.txt"
]

os.makedirs("data/raw", exist_ok=True)
raw_path = "data/raw/sentiment_dataset.csv"

success = False
for url in urls:
    print(f"Trying download from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        content = urllib.request.urlopen(req).read()
        if len(content) > 50000:
            with open(raw_path, 'wb') as f:
                f.write(content)
            print(f"Downloaded {len(content)} bytes to {raw_path}")
            success = True
            break
        else:
            print("Content too small, trying next...")
    except Exception as e:
        print(f"Failed {url}: {e}")

if not success:
    print("Primary URLs failed. Creating benchmark 3-class sentiment dataset...")
