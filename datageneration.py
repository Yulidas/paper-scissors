# generate_data.py

import pandas as pd
from RPS_game import play, abbey
import random

def strategy_player(prev_opponent_play, strategy_state):
    possible_moves = ['R', 'P', 'S']
    strategy = strategy_state['strategy']
    count = strategy_state['count']

    if strategy == 'sequence':
        # Play a fixed sequence
        moves = ['R', 'P', 'S', 'R', 'P', 'S']
        move = moves[count % len(moves)]
        strategy_state['count'] += 1
        return move
    elif strategy == 'mirror':
        # Mirror Abbey's last move
        if prev_opponent_play == '':
            return random.choice(possible_moves)
        else:
            return prev_opponent_play
    elif strategy == 'anti_mirror':
        # Play the move that beats Abbey's last move
        ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}
        if prev_opponent_play == '':
            return random.choice(possible_moves)
        else:
            return ideal_response[prev_opponent_play]
    elif strategy == 'random':
        return random.choice(possible_moves)
    else:
        return random.choice(possible_moves)

def generate_data(num_games_per_strategy):
    data = []

    strategies = ['sequence', 'mirror', 'anti_mirror', 'random']

    for strat in strategies:
        strategy_state = {'strategy': strat, 'count': 0}
        opponent_history = []
        my_history = []

        def player(prev_opponent_play):
            my_move = strategy_player(prev_opponent_play, strategy_state)
            my_history.append(my_move)

            if prev_opponent_play == '':
                prev_opponent_play = 'R'
            opponent_history.append(prev_opponent_play)

            # Record the data
            data.append({
                'strategy': strat,
                'my_second_last_move': my_history[-2] if len(my_history) > 1 else 'N',
                'my_last_move': my_history[-1],
                'opponent_second_last_move': opponent_history[-2] if len(opponent_history) > 1 else 'N',
                'opponent_last_move': opponent_history[-1],
                'opponent_next_move': 'N'  # Placeholder
            })

            return my_move

        # Simulate the games
        for _ in range(num_games_per_strategy):
            play(player, abbey, 1)

    # Update 'opponent_next_move' in data
    for i in range(len(data) - 1):
        data[i]['opponent_next_move'] = data[i + 1]['opponent_last_move']

    # Remove the last entry as we don't have the next move for it
    data = data[:-1]

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('rps_abbey_data.csv', index=False)
    print(f"Data saved to 'rps_abbey_data.csv' with {len(df)} records.")

if __name__ == "__main__":
    generate_data(10000)  # Generate data from 10,000 games per strategy
