import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


original_path = "/content/urdu-fake-news/data/Combined.csv"
cleaned_path = "/content/urdu-fake-news/data/Combined.cleaned.csv"

df_orig = pd.read_csv(original_path, encoding_errors="ignore")
df_clean = pd.read_csv(cleaned_path, encoding_errors="ignore")

if 'text' not in df_orig.columns:
    df_orig.rename(columns={df_orig.columns[0]: 'text'}, inplace=True)
if 'label' not in df_orig.columns:
    df_orig.rename(columns={df_orig.columns[-1]: 'label'}, inplace=True)

print("Original Dataset Sample")
print(df_orig[['text', 'label']].head(3))

print("Cleaned Dataset Sample")
print(df_clean[['text', 'label']].head(3))
print("Original Dataset Shape:", df_orig.shape)
print("Cleaned Dataset Shape:", df_clean.shape)
print("\nMissing Values (Original):")
print(df_orig.isnull().sum())
print("\nMissing Values (Cleaned):")
print(df_clean.isnull().sum())
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.countplot(x=df_orig['label'])
plt.title("Original Dataset Label Distribution")

plt.subplot(1,2,2)
sns.countplot(x=df_clean['label'])
plt.title("Cleaned Dataset Label Distribution")

plt.tight_layout()
plt.show()
df_orig["text_len"] = df_orig["text"].astype(str).apply(len)
df_clean["text_len"] = df_clean["text_clean"].astype(str).apply(len)


plt.figure(figsize=(12,4))


plt.subplot(1,2,1)
sns.histplot(df_orig["text_len"], bins=60, kde=True)
plt.title("Original Dataset - Text Length")
plt.xlabel("Text Length (characters)")
plt.ylabel("Frequency")
print("Original Dataset:")
print(" Mean length:", df_orig['text_len'].mean())
print(" Max length:", df_orig['text_len'].max())
print()


plt.subplot(1,2,2)
sns.histplot(df_clean["text_len"], bins=60, kde=True, color='green')
plt.title("Cleaned Dataset - Text Length")
plt.xlabel("Text Length (characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("Cleaned Dataset:")
print(" Mean length:", df_clean['text_len'].mean())
print(" Max length:", df_clean['text_len'].max())
def non_urdu_ratio(text):
    total = len(str(text))
    eng = len(re.findall(r"[A-Za-z]", str(text)))
    return eng / total if total > 0 else 0

df_orig["non_urdu_ratio"] = df_orig["text"].apply(non_urdu_ratio)
df_clean["non_urdu_ratio"] = df_clean["text_clean"].apply(non_urdu_ratio)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.histplot(df_orig["non_urdu_ratio"], bins=40)
plt.title("Original Dataset – Roman Urdu Ratio")

plt.subplot(1,2,2)
sns.histplot(df_clean["non_urdu_ratio"], bins=40)
plt.title("Cleaned Dataset – Roman Urdu Ratio")
plt.show()
def count_oov(text):
    tokens = tokenizer.tokenize(str(text))
    return tokens.count('<unk>')

df_orig["oov_count"] = df_orig["text"].apply(count_oov)
df_clean["oov_count"] = df_clean["text"].apply(count_oov)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.histplot(df_orig["oov_count"], bins=50, kde=True)
plt.title("Original - OOV Tokens")

plt.subplot(1,2,2)
sns.histplot(df_clean["oov_count"], bins=50, kde=True, color="green")
plt.title("Cleaned - OOV Tokens")
plt.show()

print("Original Dataset:")
print("Missing text:", df_orig['text'].isna().sum())
print("Missing labels:", df_orig['label'].isna().sum())

non_utf_orig = df_orig['text'].apply(lambda x: any(ord(ch) > 10000 for ch in str(x)))
print("High-Unicode / RTL characters:", non_utf_orig.sum())

print("\n")
print("Clean Dataset:")
print("Missing text:", df_clean['text'].isna().sum())
print("Missing labels:", df_clean['label'].isna().sum())

non_utf_clean = df_clean['text'].apply(lambda x: any(ord(ch) > 10000 for ch in str(x)))
print("High-Unicode / RTL characters:", non_utf_clean.sum())

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


df_orig["token_length"] = df_orig["text"].apply(lambda x: len(tokenizer.tokenize(str(x))))
print("Original Dataset (XLM-R)")
print(df_orig["token_length"].describe())

df_clean["token_length"] = df_clean["text"].apply(lambda x: len(tokenizer.tokenize(str(x))))
print("\nCleaned Dataset  (XLM-R)")
print(df_clean["token_length"].describe())

df_orig['token_count'] = df_orig['text'].apply(lambda x: len(tokenizer.tokenize(str(x))))
print(df_orig['token_count'].describe())

df_clean['token_count'] = df_clean['text'].apply(lambda x: len(tokenizer.tokenize(str(x))))
print(df_clean['token_count'].describe())

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
sns.histplot(df_orig['token_count'], bins=50, kde=True, color="#1f77b4")
plt.title("Original Dataset (XLM-R)", fontsize=13)
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
sns.histplot(df_clean['token_count'], bins=50, kde=True, color="#2ca02c")
plt.title("Cleaned Dataset (XLM-R)", fontsize=13)
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")

plt.suptitle("Token Count Comparison Original vs Cleaned Dataset", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
from collections import Counter
import regex as re
from IPython.display import display, HTML


def urdu_tokens(text):
    return re.findall(r"[\u0600-\u06FF]+", str(text))


orig_words = [w for text in df_orig['text'] for w in urdu_tokens(text)]
orig_common = Counter(orig_words).most_common(15)

print("\nTop Urdu Words in Original Dataset:")
display(pd.DataFrame(orig_common, columns=["Word", "Frequency"]))

print("\nRTL Sample from Original Dataset:")
sample_text = df_orig['text'].iloc[0]
display(HTML(f"<p style='direction: rtl; font-size:16px'>{sample_text}</p>"))


clean_words = [w for text in df_clean['text'] for w in urdu_tokens(text)]
clean_common = Counter(clean_words).most_common(15)

print("\n Top Urdu Words in Cleaned Dataset:")
display(pd.DataFrame(clean_common, columns=["Word", "Frequency"]))

print("\n RTL Sample from Cleaned Dataset:")
sample_text_clean = df_clean['text'].iloc[0]
display(HTML(f"<p style='direction: rtl; font-size:16px'>{sample_text_clean}</p>"))


print("Original Dataset Summary ")
print(f"Total rows analyzed: {len(df_orig)}")
print("Label balance:", df_orig['label'].value_counts().to_dict())
print("Average text length:", round(df_orig['text_len'].mean(), 2))
print("Avg token count (XLM-R):", round(df_orig['token_count'].mean(), 2))
print("Avg OOV tokens:", round(df_orig['oov_count'].mean(), 2))
print("Avg non-Urdu ratio:", round(df_orig['non_urdu_ratio'].mean(), 3))


print("\nCleaned Dataset Summary ")
print(f"Total rows analyzed: {len(df_clean)}")
print("Label balance:", df_clean['label'].value_counts().to_dict())
print("Average text length:", round(df_clean['text_len'].mean(), 2))
print("Avg token count (XLM-R):", round(df_clean['token_count'].mean(), 2))
print("Avg OOV tokens:", round(df_clean['oov_count'].mean(), 2))
print("Avg non-Urdu ratio:", round(df_clean['non_urdu_ratio'].mean(), 3))
