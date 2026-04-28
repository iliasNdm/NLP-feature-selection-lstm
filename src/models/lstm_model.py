from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, Dense, Dropout,
                                     Bidirectional, GlobalMaxPooling1D,
                                     BatchNormalization)
from tensorflow.keras.optimizers import Adam


def build_lstm_model(vocab_size, maxlen, n_classes, embedding_dim=128,
                     lstm_units=64, dropout=0.2):
    """BiLSTM + GlobalMaxPooling — architecture principale du projet"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                  input_length=maxlen),
        Bidirectional(LSTM(lstm_units, return_sequences=True,
                           dropout=dropout, recurrent_dropout=dropout)),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model