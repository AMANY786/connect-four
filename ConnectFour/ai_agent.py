"""
This is the code that implements the AI agent to predict the next best move per each turn
"""

import random
import numpy as np

# Constants
ROW_COUNT = 6
COLUMN_COUNT = 7
AI_PLAYER = 2
HUMAN_PLAYER = 1
EMPTY = 0
WINDOW_LENGTH = 4

# Function to check valid moves
def is_valid_location(board, col):
    return board[0][col] == 0

# Get the next available row in a column
def get_next_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r][col] == EMPTY:
            return r

# Drop a piece in the board
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Check if a move results in a win
def winning_move(board, piece):
    # Horizontal win
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r][c + i] == piece for i in range(WINDOW_LENGTH)):
                return True
    
    # Vertical win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all(board[r + i][c] == piece for i in range(WINDOW_LENGTH)):
                return True
    
    # Positive diagonal win
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r + i][c + i] == piece for i in range(WINDOW_LENGTH)):
                return True
    
    # Negative diagonal win
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            if all(board[r - i][c + i] == piece for i in range(WINDOW_LENGTH)):
                return True
    
    return False

# Score the board for AI decision making
def evaluate_window(window, piece):
    score = 0
    opponent_piece = HUMAN_PLAYER if piece == AI_PLAYER else AI_PLAYER

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 10
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 5
    
    if window.count(opponent_piece) == 3 and window.count(EMPTY) == 1:
        score -= 80
    
    return score

# Get the total score of the board
def score_position(board, piece):
    score = 0
    
    # Center column preference
    center_array = [board[r][COLUMN_COUNT // 2] for r in range(ROW_COUNT)]
    score += center_array.count(piece) * 6
    
    # Score horizontal
    for r in range(ROW_COUNT):
        row_array = [board[r][c] for c in range(COLUMN_COUNT)]
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window(row_array[c:c + WINDOW_LENGTH], piece)
    
    # Score vertical
    for c in range(COLUMN_COUNT):
        col_array = [board[r][c] for r in range(ROW_COUNT)]
        for r in range(ROW_COUNT - 3):
            score += evaluate_window(col_array[r:r + WINDOW_LENGTH], piece)
    
    # Score positive diagonals
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)
    
    # Score negative diagonals
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)
    
    return score

# Minimax function with alpha-beta pruning
def minimax(board, depth, alpha, beta, maximizing_player):
    valid_columns = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
    is_terminal = winning_move(board, AI_PLAYER) or winning_move(board, HUMAN_PLAYER) or len(valid_columns) == 0
    
    if depth == 0 or is_terminal:
        if winning_move(board, AI_PLAYER):
            return (None, 1000000)
        elif winning_move(board, HUMAN_PLAYER):
            return (None, -1000000)
        else:
            return (None, score_position(board, AI_PLAYER))
    
    if maximizing_player:
        value = -np.inf
        best_col = random.choice(valid_columns)
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = [row[:] for row in board]
            drop_piece(temp_board, row, col, AI_PLAYER)
            new_score = minimax(temp_board, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = np.inf
        best_col = random.choice(valid_columns)
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = [row[:] for row in board]
            drop_piece(temp_board, row, col, HUMAN_PLAYER)
            new_score = minimax(temp_board, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

# AI chooses the best move
def get_best_move(board, depth=4):
    best_col, _ = minimax(board, depth, -np.inf, np.inf, True)
    return best_col
