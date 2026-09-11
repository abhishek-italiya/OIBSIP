import os
import urllib.request
import pandas as pd

os.makedirs("data/raw", exist_ok=True)
raw_path = "data/raw/creditcard.csv"

urls = [
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
    "https://media.githubusercontent.com/media/gregversteeg/bio_viz/master/data/creditcard.csv",
    "https://raw.githubusercontent.com/nvduran/dataset_creditcard/main/creditcard.csv"
]

success = False
for url in urls:
    print(f"Trying to download dataset from: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(raw_path, 'wb') as out_file:
            data = response.read()
            if len(data) > 1000000: # Ensure valid file (> 1MB)
                out_file.write(data)
                print(f"Successfully downloaded {len(data):,} bytes to {raw_path}")
                success = True
                break
    except Exception as e:
        print(f"Failed to download from {url}: {e}")

if success:
    df = pd.read_csv(raw_path)
    print("\n=== DATASET VERIFICATION ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]} ... {df.columns.tolist()[-3:]}")
    print(f"Class counts:\n{df['Class'].value_counts()}")
else:
    print("Failed to download dataset from all sources.")
