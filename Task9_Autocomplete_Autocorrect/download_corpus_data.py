import os
import nltk

def download_corpus():
    """
    Downloads and extracts the public-domain Herman Melville 'Moby Dick' corpus from NLTK Project Gutenberg.
    Saves raw text to data/raw/corpus.txt.
    """
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filepath = os.path.join(raw_dir, "corpus.txt")
    
    print("=== DOWNLOADING PUBLIC DOMAIN GUTENBERG CORPUS ===")
    nltk.download('gutenberg', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    from nltk.corpus import gutenberg
    text = gutenberg.raw('melville-moby_dick.txt')
    
    with open(raw_filepath, 'w', encoding='utf-8') as f:
        f.write(text)
        
    print(f"Corpus successfully saved to: {raw_filepath}")
    print(f"Total Character Count : {len(text):,}")

if __name__ == "__main__":
    download_corpus()
