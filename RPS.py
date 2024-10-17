# RPS.py
import random
import numpy as np
from tensorflow.keras.models import load_model
import joblib

def player(prev_opponent_play, opponent_history=[], my_history=[], strategy=[''], count=[0], model_data={}):
    # Initialize possible moves and counters
    possible_moves = ['R', 'P', 'S']
    ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}

    # Record the opponent's and our own moves
    if prev_opponent_play == '':
        prev_opponent_play = random.choice(possible_moves)
    opponent_history.append(prev_opponent_play)

    # Determine strategy
    count[0] += 1
    if count[0] < 5:
        # First few moves, play randomly
        my_move = random.choice(possible_moves)
        my_history.append(my_move)
        strategy[0] = 'random'  # Start with random strategy
        return my_move

    # Detect Quincy
    quincy_sequence = ['R', 'R', 'P', 'P', 'S']
    if detect_quincy(opponent_history, quincy_sequence):
        strategy[0] = 'sequence'

    # Detect Kris
    elif detect_kris(opponent_history, my_history, ideal_response):
        strategy[0] = 'mirror'

    # Detect Mrugesh
    elif detect_mrugesh(opponent_history):
        strategy[0] = 'anti_mirror'

    # Use Abbey counter if no other strategy is detected
    else:
        strategy[0] = 'abbey'

    # Apply the strategy
    if strategy[0] == 'abbey':
        my_move = counter_abbey(opponent_history, my_history, strategy, model_data)
    else:
        my_move = strategy_player(prev_opponent_play, {'strategy': strategy[0], 'count': count[0]})

    my_history.append(my_move)
    return my_move

def detect_quincy(opponent_history, quincy_sequence):
    # Check if opponent's moves match Quincy's sequence
    seq_len = len(quincy_sequence)
    if len(opponent_history) < seq_len:
        return False
    last_moves = opponent_history[-seq_len:]
    return last_moves == quincy_sequence

def counter_quincy(opponent_history, quincy_sequence, ideal_response):
    # Predict Quincy's next move and counter it
    seq_len = len(quincy_sequence)
    idx = len(opponent_history) % seq_len
    predicted_move = quincy_sequence[idx]
    return ideal_response[predicted_move]

def detect_kris(opponent_history, my_history, ideal_response):
    # Kris plays the move that beats our last move
    if len(my_history) < 1:
        return False
    return opponent_history[-1] == ideal_response[my_history[-1]]

def counter_kris(my_history):
    # Play the move that loses to our last move
    if len(my_history) < 1:
        return random.choice(['R', 'P', 'S'])
    losing_move = {'R': 'S', 'P': 'R', 'S': 'P'}
    return losing_move[my_history[-1]]

def detect_mrugesh(opponent_history):
    # Since we cannot directly detect Mrugesh, assume default strategy
    return False  # For simplicity

def counter_mrugesh(my_history, ideal_response):
    # Keep track of our own moves to manipulate frequency
    if len(my_history) < 10:
        most_common_move = 'R'
    else:
        last_ten = my_history[-10:]
        most_common_move = max(set(last_ten), key=last_ten.count)
    # Mrugesh will play the move that beats our most common move
    mrugesh_move = ideal_response[most_common_move]
    # We play the move that beats Mrugesh's expected move
    return ideal_response[mrugesh_move]

def counter_abbey(opponent_history, my_history, strategy, model_data):
    # Use the trained machine learning model to predict Abbey's next move
    if not model_data:
        # Load model and encoders
        model_data['model'] = load_model('rps_abbey_model.h5')
        model_data['move_encoder'] = joblib.load('move_encoder.pkl')
        model_data['strategy_encoder'] = joblib.load('strategy_encoder.pkl')

    move_encoder = model_data['move_encoder']
    strategy_encoder = model_data['strategy_encoder']
    model = model_data['model']

    # Prepare input sequence
    my_second_last_move = my_history[-2] if len(my_history) > 1 else 'N'
    my_last_move = my_history[-1] if my_history else 'N'
    opponent_second_last_move = opponent_history[-2] if len(opponent_history) > 1 else 'N'
    opponent_last_move = opponent_history[-1]

    # Map 'abbey' to 'random' for encoding
    strategy_label = strategy[0]
    if strategy_label == 'abbey':
        strategy_label = 'random'  # or another strategy from ['sequence', 'mirror', 'anti_mirror', 'random']

    input_sequence = [
        move_encoder.transform([my_second_last_move])[0],
        move_encoder.transform([my_last_move])[0],
        move_encoder.transform([opponent_second_last_move])[0],
        move_encoder.transform([opponent_last_move])[0],
        strategy_encoder.transform([strategy_label])[0]
    ]

    input_sequence = np.array(input_sequence).reshape(1, -1)

    # Predict Abbey's next move
    prediction = model.predict(input_sequence, verbose=0)
    predicted_move_index = np.argmax(prediction)
    predicted_move = move_encoder.inverse_transform([predicted_move_index])[0]

    if predicted_move == 'N':
        predicted_move = random.choice(['R', 'P', 'S'])

    # Choose the move that beats Abbey's predicted move
    ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}
    my_move = ideal_response.get(predicted_move, random.choice(['R', 'P', 'S']))

    return my_move

def strategy_player(prev_opponent_play, strategy_state):
    # Same as in data generation
    possible_moves = ['R', 'P', 'S']
    strategy = strategy_state['strategy']
    count = strategy_state['count']

    if strategy == 'sequence':
        moves = ['R', 'P', 'S', 'R', 'P', 'S']
        move = moves[count % len(moves)]
        strategy_state['count'] += 1
        return move
    elif strategy == 'mirror':
        if prev_opponent_play == '':
            return random.choice(possible_moves)
        else:
            return prev_opponent_play
    elif strategy == 'anti_mirror':
        ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}
        if prev_opponent_play == '':
            return random.choice(possible_moves)
        else:
            return ideal_response[prev_opponent_play]
    elif strategy == 'random':
        return random.choice(possible_moves)
    else:
        return random.choice(possible_moves)