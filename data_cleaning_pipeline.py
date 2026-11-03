"""
Beauty & Wellness Customer Sentiment & Feedback Analysis
Week 3: End-to-End Data Cleaning & Preprocessing Pipeline
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 1. DOMAIN-AWARE STOPWORD DEFINITION
# ---------------------------------------------------------
BASE_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'he', 'she', 'it',
    'they', 'the', 'a', 'an', 'and', 'or', 'because', 'as', 'until', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through'
}

# General e-commerce filler words that carry no sentiment polarity
DOMAIN_FILLER = {'product', 'item', 'skin', 'beauty', 'sephora', 'bottle', 'purchase', 'bought'}

# CRITICAL: Preserve negation terms so sentiment inversion is not lost
NEGATION_TOKENS = {'not', 'no', 'never', 'neither', 'hardly', 'scarcely', 'barely', 'without'}

EFFECTIVE_STOPWORDS = (BASE_STOPWORDS | DOMAIN_FILLER) - NEGATION_TOKENS

def clean_raw_text(text):
    """
    Sanitizes raw unstructured customer review text:
    - Strips HTML tags, hyperlinks, and non-alphabetical noise
    - Lowers character case
    - Filters filler stopwords while retaining sentiment negation terms
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Remove HTML tags & hyperlinks
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # 2. Retain alphabetical tokens only
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 3. Tokenize and filter
    tokens = [w.lower() for w in text.split() if len(w) > 2]
    filtered_tokens = [w for w in tokens if w not in EFFECTIVE_STOPWORDS]
    
    return ' '.join(filtered_tokens)

def execute_cleaning_pipeline(data_path=None):
    """
    Executes the 5-Gate Preprocessing Pipeline:
    Gate 1: Ingestion & Duplicate Handling
    Gate 2: Null Value Resolution (Demographic & Numerical)
    Gate 3: Outlier & Noise Filtering (IQR on Word Count)
    Gate 4: Text Normalization
    Gate 5: Feature Normalization & Quality Audit Checksum
    """
    print("=" * 70)
    print("Executing Beauty & Wellness Data Preprocessing Pipeline (Week 3)")
    print("=" * 70)
    
    # Load data or construct representative validation batch
    if data_path:
        df = pd.read_csv(data_path)
    else:
        print("[INFO] No external dataset path provided. Ingesting validation batch...")
        sample_batch = [
            {"review_id": 101, "product_id": "P428230", "rating": 1.0, "skin_type": "Sensitive", "price_usd": 48.0,
             "review_text": "Gave me terrible cystic acne along my jawline within 4 days. Not good!"},
            {"review_id": 102, "product_id": "P428230", "rating": 1.0, "skin_type": np.nan, "price_usd": 48.0,
             "review_text": "Burning sensation on cheeks. Pump broke after 2 uses.<br/>Very cheap packaging."},
            {"review_id": 103, "product_id": "P981244", "rating": 5.0, "skin_type": "Dry", "price_usd": 64.0,
             "review_text": "Holy grail moisturizer! Absorbs instantly and restores damaged skin barrier."},
            {"review_id": 104, "product_id": "P981244", "rating": 5.0, "skin_type": "Normal", "price_usd": np.nan,
             "review_text": "Repurchased 3 times. Excellent winter hydration without feeling greasy."},
            {"review_id": 105, "product_id": "P112349", "rating": 2.0, "skin_type": "Combination", "price_usd": 24.0,
             "review_text": "Leaves oily residue and clogged pores. Strong artificial perfume."},
            {"review_id": 106, "product_id": "P112349", "rating": 4.0, "skin_type": np.nan, "price_usd": 24.0,
             "review_text": "Gentle daily foaming cleanser. Decent value for money."},
            {"review_id": 107, "product_id": "P999999", "rating": 5.0, "skin_type": "Oily", "price_usd": 15.0,
             "review_text": "ok"},  # Short noise outlier (< 3 words)
            {"review_id": 108, "product_id": "P999999", "rating": 5.0, "skin_type": "Oily", "price_usd": 15.0,
             "review_text": "ok"}   # Exact duplicate review
        ]
        df = pd.DataFrame(sample_batch)
    
    raw_count = len(df)
    print(f"[GATE 1] Raw Records Ingested: {raw_count}")
    
    # 1. Duplicate & Missing Ground-Truth Cleanup
    df.drop_duplicates(subset=['review_text'], inplace=True)
    df.dropna(subset=['review_text', 'rating'], inplace=True)
    print(f"[GATE 1] After Deduplication & Primary Null Removal: {len(df)} records")
    
    # 2. Demographic & Numerical Imputation
    if 'skin_type' in df.columns:
        df['skin_type'] = df['skin_type'].fillna('Not Specified')
    if 'price_usd' in df.columns:
        median_price = df['price_usd'].median()
        df['price_usd'] = df['price_usd'].fillna(median_price)
    print("[GATE 2] Imputed missing 'skin_type' as 'Not Specified' & 'price_usd' with median value")
    
    # 3. Text Sanitization
    df['cleaned_text'] = df['review_text'].apply(clean_raw_text)
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    
    # 4. Outlier Filtering (IQR Method on Word Count)
    q25 = df['word_count'].quantile(0.25)
    q75 = df['word_count'].quantile(0.75)
    iqr = q75 - q25
    upper_bound = q75 + (2.5 * iqr)
    
    # Filter short micro-reviews (< 3 words) and promotional spam (> upper_bound)
    clean_df = df[(df['word_count'] >= 3) & (df['word_count'] <= upper_bound)].copy()
    print(f"[GATE 3] Outlier Filter Applied (IQR Bound: 3 to {upper_bound:.1f} words). Remaining: {len(clean_df)} records")
    
    # 5. Rating Clipping & MinMax Scaling
    clean_df['rating'] = clean_df['rating'].clip(lower=1.0, upper=5.0)
    scaler = MinMaxScaler()
    clean_df['norm_word_count'] = scaler.fit_transform(clean_df[['word_count']])
    print("[GATE 4 & 5] Rating clipped to [1.0, 5.0]. Word count normalized via MinMaxScaler.")
    
    print("\n[AUDIT SUMMARY]")
    print(f"Total Rows Ingested: {raw_count}")
    print(f"Clean Analytical Rows: {len(clean_df)}")
    print(f"Data Retention Rate: {(len(clean_df)/raw_count)*100:.1f}%")
    print("\nSample Cleaned Output:")
    print(clean_df[['rating', 'skin_type', 'cleaned_text', 'word_count']].head(3))
    
    return clean_df

if __name__ == "__main__":
    execute_cleaning_pipeline()
