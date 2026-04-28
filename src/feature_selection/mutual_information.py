from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def compute_mi_scores(X_tfidf, y, feature_names):
    """Calcule les scores MI pour chaque feature."""
    mi_scores = mutual_info_classif(X_tfidf, y, random_state=42)
    mi_dict = dict(zip(feature_names, mi_scores))
    mi_sorted = dict(sorted(mi_dict.items(), key=lambda x: x[1], reverse=True))
    return mi_sorted

def select_top_k_features(mi_scores, k):
    """Retourne les top-k features selon MI."""
    return list(mi_scores.keys())[:k]

def rebuild_sequences(X_train, X_val, X_test, top_features, maxlen):
    """Reconstruit les séquences Keras avec un vocabulaire réduit."""
    vocab = set(top_features)

    def filter_text(text):
        return ' '.join([w for w in str(text).split() if w in vocab])

    X_train_f = X_train.apply(filter_text)
    X_val_f   = X_val.apply(filter_text)
    X_test_f  = X_test.apply(filter_text)

    tokenizer = Tokenizer(num_words=len(top_features) + 1, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_f)

    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_f),
                                maxlen=maxlen, padding='post', truncating='post')
    X_val_pad   = pad_sequences(tokenizer.texts_to_sequences(X_val_f),
                                maxlen=maxlen, padding='post', truncating='post')
    X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test_f),
                                maxlen=maxlen, padding='post', truncating='post')

    return X_train_pad, X_val_pad, X_test_pad, tokenizer