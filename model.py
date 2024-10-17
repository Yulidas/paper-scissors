# train_model.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model():
    # Load the data
    df = pd.read_csv('rps_abbey_data.csv')

    # Fill missing values with 'N'
    df.fillna('N', inplace=True)

    # Define mappings
    moves = ['R', 'P', 'S', 'N']
    strategies = ['sequence', 'mirror', 'anti_mirror', 'random', 'abbey']

    move_encoder = LabelEncoder()
    move_encoder.fit(moves)
    strategy_encoder = LabelEncoder()
    strategy_encoder.fit(strategies)

    # Save encoders
    joblib.dump(move_encoder, 'move_encoder.pkl')
    joblib.dump(strategy_encoder, 'strategy_encoder.pkl')

    # Prepare input features
    X_moves = df[['my_second_last_move', 'my_last_move', 'opponent_second_last_move', 'opponent_last_move']]
    X_moves = X_moves.applymap(lambda x: move_encoder.transform([x])[0])

    # Shift strategy indices to avoid overlap with move indices
    X_strategy = df['strategy'].apply(lambda x: strategy_encoder.transform([x])[0] + len(move_encoder.classes_))

    # Combine features
    X = np.concatenate([X_moves.values, X_strategy.values.reshape(-1, 1)], axis=1)

    # Prepare target variable
    y = df['opponent_next_move'].apply(lambda x: move_encoder.transform([x])[0])
    y_encoded = to_categorical(y, num_classes=len(move_encoder.classes_))

    # Build the model
    total_vocab_size = len(move_encoder.classes_) + len(strategy_encoder.classes_)
    model = Sequential()
    model.add(Embedding(input_dim=total_vocab_size, output_dim=10, input_length=5))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(move_encoder.classes_), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y_encoded, epochs=20, batch_size=128, validation_split=0.1)

    # Save the model
    model.save('rps_abbey_model.h5')
    print("Model and encoders saved.")

if __name__ == "__main__":
    train_model()
