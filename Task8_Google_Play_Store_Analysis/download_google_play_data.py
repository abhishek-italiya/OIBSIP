import os
import urllib.request
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

dataset_files = {
    "googleplaystore.csv": [
        "https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/master/googleplaystore.csv",
        "https://raw.githubusercontent.com/Sudz24/PlayStore/master/googleplaystore.csv"
    ],
    "googleplaystore_user_reviews.csv": [
        "https://raw.githubusercontent.com/akhil2994/Play-Store-App-Popularity-Prediction/master/googleplaystore_user_reviews.csv"
    ]
}

print("=== DOWNLOADING GOOGLE PLAY STORE DATASETS ===")

for filename, urls in dataset_files.items():
    file_path = os.path.join("data/raw", filename)
    success = False
    for url in urls:
        print(f"Downloading {filename} from: {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                data = response.read()
                if len(data) > 10000:
                    out_file.write(data)
                    print(f"Successfully downloaded {filename} ({len(data):,} bytes)")
                    success = True
                    break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            
    if not success:
        print(f"Warning: Could not download {filename} automatically.")

# Verify downloaded datasets
print("\n=== VERIFYING DATASETS ===")
try:
    df_apps = pd.read_csv("data/raw/googleplaystore.csv")
    print(f"Apps Dataset: {df_apps.shape[0]:,} rows x {df_apps.shape[1]} columns")
    print(f"Apps Columns: {df_apps.columns.tolist()}")
except Exception as e:
    print(f"Error reading googleplaystore.csv: {e}")

try:
    df_reviews = pd.read_csv("data/raw/googleplaystore_user_reviews.csv")
    print(f"User Reviews Dataset: {df_reviews.shape[0]:,} rows x {df_reviews.shape[1]} columns")
    print(f"Reviews Columns: {df_reviews.columns.tolist()}")
except Exception as e:
    print(f"Error reading googleplaystore_user_reviews.csv: {e}")
