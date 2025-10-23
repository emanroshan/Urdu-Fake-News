import pandas as pd
import re
import unicodedata
from pathlib import Path

DATA_PATH = Path("/content/urdu-fake-news/data/Combined.csv")
df = pd.read_csv(DATA_PATH, engine="python", encoding_errors="ignore")
print("Before cleaning:", len(df))


label_col = [c for c in df.columns if "label" in c.lower()][0]
df[label_col] = df[label_col].astype(str).str.strip().str.title()
df = df[df[label_col].isin(["True", "Fake"])]

text_col = [c for c in df.columns if "news" in c.lower() or "text" in c.lower()][0]
df[text_col] = df[text_col].astype(str).str.strip()
df = df[df[text_col].str.len() > 25]
df = df.drop_duplicates(subset=[text_col])
df = df.rename(columns={text_col: "text", label_col: "label"})


# Unicode Normalization
def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)

# Remove Zero-Width Characters
ZERO_WIDTH = ["\u200B", "\u200C", "\u200D", "\uFEFF", "\u2060"]
def remove_zero_width(text):
    for zw in ZERO_WIDTH:
        text = text.replace(zw, "")
    return text

# Normalize Urdu punctuation + digits
PUNCT_MAP = str.maketrans({"،": ",", "۔": ".", "؛": ";", "؟": "?", "’": "'", "‘": "'", "“": '"', "”": '"', "ـ": ""})
DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
def normalize_text(text):
    return text.translate(PUNCT_MAP).translate(DIGIT_MAP)

# Remove Diacritics
def remove_diacritics(text):
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

# Clean whitespace 
def clean_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

# Sanitize URLs/emails/media
def sanitize_urls(text):
    text = re.sub(r"(https?://\S+|www\.\S+)", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    return text

# Flag Urdu vs Roman Urdu content
def detect_language(text):
    urdu = sum('\u0600' <= ch <= '\u06FF' for ch in text)
    roman = sum('a' <= ch.lower() <= 'z' for ch in text)
    return "urdu" if urdu > roman else "roman" if roman > urdu else "mixed"

# Tokenization 
def simple_tokenize(text):
    return text.split()

# Full cleaning function
def preprocess(text):
    text = normalize_unicode(text)
    text = remove_zero_width(text)
    text = sanitize_urls(text)
    text = normalize_text(text)
    text = remove_diacritics(text)
    text = clean_whitespace(text)
    return text

df["text_clean"] = df["text"].apply(preprocess)
df["lang_flag"] = df["text_clean"].apply(detect_language)
df["tokens"] = df["text_clean"].apply(simple_tokenize)

try:
    from transformers import XLMRobertaTokenizerFast
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    def xlmr_truncate(text):
        ids = tokenizer.encode(str(text), truncation=True, max_length=512)
        return tokenizer.decode(ids)

    df["xlmr_trunc"] = df["text_clean"].apply(xlmr_truncate)
    print("\nXLM-R transformer column added: 'xlmr_trunc'")
except:
    df["xlmr_trunc"] = None
    print("\ntransformers not installed, skipping XLM-R compatibility")


OUTPUT_PATH = "/content/urdu-fake-news/data/Combined.cleaned.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nRows after cleaning: {len(df)}")
print(f"File saved to: {OUTPUT_PATH}")
print("\nSample preview:")
print(df[['text_clean', 'lang_flag', 'tokens', 'xlmr_trunc']].head(3))
