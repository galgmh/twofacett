# ======================================================
# DNA-GPT core detection (official logic, simplified)
# Author: adapted from NEC Labs DNA-GPT (Yang et al. 2023)
# ======================================================

# !pip install openai nltk rouge_score spacy
# !python -m spacy download en_core_web_sm

import re, six, nltk, spacy, numpy as np
from rouge_score.rouge_scorer import _create_ngrams
from nltk.stem.porter import PorterStemmer
from openai import OpenAI

# ==== configurations ====
API_KEY = "" # your OpenAI API key here
client = OpenAI(api_key=API_KEY)
temperature = 0.7
max_new_tokens = 300
regen_number = 30
truncate_ratio = 0.5
threshold = 0.00025  # default from official code


stemmer = PorterStemmer()
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words
nltk.download('punkt')

# ---------- utility functions ----------
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))
    tokens = re.split(r"\s+", text)
    tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens if x not in stopwords]
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]
    return tokens


def get_ngram_score(t1, t2, n):
    g1 = _create_ngrams(t1, n)
    g2 = _create_ngrams(t2, n)
    inter = 0
    for k in g1:
        inter += min(g1[k], g2[k])
    return inter / max(sum(g1.values()), 1)


def N_gram_detector(ngram_scores):
    score = 0
    non_zero = []
    for idx, val in enumerate(ngram_scores, start=1):
        if idx <= 3:
            continue
        score += idx * np.log(idx) * val
        if val != 0:
            non_zero.append(idx)
    return score / (sum(non_zero) + 1e-8)


# ---------- main detection ----------
def dna_gpt_score(text, model="gpt-3.5-turbo-instruct"):
    # 1. truncate
    words = text.split()
    prefix = " ".join(words[: int(len(words) * truncate_ratio)])
    suffix = " ".join(words[int(len(words) * truncate_ratio):])

    # 2. regen with GPT
    regen = client.completions.create(
        model=model,
        prompt=prefix,
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=regen_number
    )

    suffix_tokens = tokenize(suffix)
    scores = []

    for choice in regen.choices:
        regen_text = choice.text.strip()
        regen_tokens = tokenize(regen_text)
        ngram_scores = [get_ngram_score(suffix_tokens, regen_tokens, n) for n in range(1, 26)]
        score = N_gram_detector(ngram_scores)
        scores.append(score)

    avg_score = np.mean(scores)
    return avg_score


def detect_text(text, model="gpt-3.5-turbo-instruct"):
    score = dna_gpt_score(text, model=model)
    return {
        "score": score,
        "label": "AI-generated" if score > threshold else "Human"
    }
