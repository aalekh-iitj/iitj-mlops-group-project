import gzip, json, pickle, random, requests
from transformers import DistilBertTokenizerFast
from utils import label2id, MyDataset

GENRE_URL_DICT = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":               "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}
PICKLE_PATH = "genre_reviews_dict.pickle"

def load_reviews(url, head=10000, sample_size=2000):
    reviews, count = [], 0
    response = requests.get(url, stream=True)
    print(f"  HTTP {response.status_code}")
    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))

def get_genre_reviews(sample_size=2000, use_cache=True):
    if use_cache:
        try:
            data = pickle.load(open(PICKLE_PATH, "rb"))
            print("Loaded from cache")
            return data
        except FileNotFoundError:
            print("No cache — downloading...")
    genre_reviews_dict = {}
    for genre, url in GENRE_URL_DICT.items():
        print(f"Loading: {genre}")
        genre_reviews_dict[genre] = load_reviews(url, head=10000, sample_size=sample_size)
    pickle.dump(genre_reviews_dict, open(PICKLE_PATH, "wb"))
    return genre_reviews_dict

def split_data(genre_reviews_dict, reviews_per_genre=1000, train_ratio=0.8):
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(reviews_per_genre, len(reviews)))
        split   = int(len(sampled) * train_ratio)
        for r in sampled[:split]:
            train_texts.append(r); train_labels.append(genre)
        for r in sampled[split:]:
            test_texts.append(r);  test_labels.append(genre)
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    return train_texts, train_labels, test_texts, test_labels

def build_datasets(train_texts, train_labels, test_texts, test_labels,
                   model_name="distilbert-base-cased", max_length=512):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_enc  = tokenizer(test_texts,  truncation=True, padding=True, max_length=max_length)
    train_ds  = MyDataset(train_enc, [label2id[y] for y in train_labels])
    test_ds   = MyDataset(test_enc,  [label2id[y] for y in test_labels])
    return train_ds, test_ds, tokenizer

def load_and_prepare(model_name="distilbert-base-cased", max_length=512,
                     reviews_per_genre=1000, use_cache=True):
    genre_reviews_dict = get_genre_reviews(use_cache=use_cache)
    train_texts, train_labels, test_texts, test_labels = split_data(
        genre_reviews_dict, reviews_per_genre=reviews_per_genre)
    return build_datasets(train_texts, train_labels, test_texts, test_labels,
                          model_name=model_name, max_length=max_length), (test_texts, test_labels)
