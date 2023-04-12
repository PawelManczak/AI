from random import choice
from copy import deepcopy

from connect4 import Connect4


def advanced_static_eval(board: Connect4, my_token: str) -> float:
    op = ""
    if my_token == 'x':
        op = 'o'
    else:
        op = "x"

    counter = 0
    op_counter = 0
    pb = board.iter_fours()
    for x in pb:
        if x.count(my_token) == 3 and x.count(op) == 0:
            counter += 3

    pb = board.iter_fours()
    for x in pb:
        if x.count(op) == 3 and x.count(my_token) == 0:
            op_counter += 3

    pb = board.iter_fours()
    for x in pb:
        if x.count(my_token) == 3 and x.count(op) == 1:
            counter += 1

    pb = board.iter_fours()
    for x in pb:
        if x.count(op) == 3 and x.count(my_token) == 1:
            op_counter += 1

    pb = board.center_column()
    for x in pb:
        counter += x.count(my_token) / 10
        op_counter += x.count(op) / 10

    first_row = board.board[0]

    op_counter -= first_row.count(op)*0.4
    counter -= first_row.count(my_token)*0.4

    pb = board.iter_fours()
    for x in pb:
        if x.count(my_token) == 2 and x.count(op) == 0:
            counter -= 0.3

    pb = board.iter_fours()
    for x in pb:
        if x.count(op) == 2 and x.count(my_token) == 0:
            op_counter -= 0.3

    return counter - op_counter


def basic_static_eval(board: Connect4, my_token: str) -> float:
    op = ""
    if my_token == 'x':
        op = 'o'
    else:
        op = "x"

    counter = 0
    op_counter = 0
    pb = board.iter_fours()
    for x in pb:
        if x.count(my_token) == 3:
            counter += 1

    pb = board.iter_fours()
    for x in pb:
        if x.count(op) == 3:
            op_counter += 1

    return counter - op_counter


class AlphaBetaAgent:

    def __init__(self, my_token='o', eval_func=basic_static_eval, max_depth=4):
        self.my_token = my_token
        self.eval_func = eval_func
        self.max_depth = max_depth

    def decide(self, board):
            possible_moves = board.possible_drops()
            if len(possible_moves) == 1:
                return possible_moves[0]

            new_board = deepcopy(board)
            score, move = self._minimax(new_board, self.max_depth, float('-inf'), float('inf'), False)

            return move

    def _minimax(self, board, depth, alpha, beta, maximizing_player):
            if depth == 0 or board.game_over:
                if board.wins == self.my_token:
                    return 1, 0
                elif board.wins is not None:
                    return -1, 0
                else:
                    return self.eval_func(board, self.my_token), 0

            if maximizing_player:
                max_score = float('-inf')
                best_move = 0
                for move in board.possible_drops():
                    new_board = deepcopy(board)
                    new_board.drop_token(move)
                    score, _ = self._minimax(new_board, depth - 1, alpha, beta, False)

                    if score > max_score:
                        best_move = move
                        max_score = score

                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
                return max_score, best_move
            else: 
                min_score = float('inf')
                best_move = 0
                for move in board.possible_drops():
                    new_board = deepcopy(board)
                    new_board.drop_token(move)
                    score, _ = self._minimax(new_board, depth - 1, alpha, beta, True)

                    if score < min_score:
                        best_move = move
                        min_score = score

                    beta = min(beta, score)
                    if beta <= alpha:
                        break
                return min_score, best_move






