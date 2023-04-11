from random import choice
from copy import deepcopy


class MinMaxAgent:
    def __init__(self, my_token, max_depth=4):
        self.my_token = my_token
        self.max_depth = max_depth

    def decide(self, board):
        possible_moves = board.possible_drops()
        if len(possible_moves) == 1:
            return possible_moves[0]
        scores = []
        for move in possible_moves:
            new_board = deepcopy(board)
            new_board.drop_token(move)
            score = self._minimax(new_board, self.max_depth, False)
            scores.append(score)
        best_score = max(scores)
        best_moves = [move for move, score in zip(possible_moves, scores) if score == best_score]
        return choice(best_moves)

    def _minimax(self, board, depth, maximizing_player):
        if depth == 0 or board.game_over:
            if board.wins == self.my_token:
                return 1
            elif board.wins is not None:
                return -1
            else:
                return self._score(board)

        if maximizing_player:
            max_score = float('-inf')
            for move in board.possible_drops():
                new_board = deepcopy(board)
                new_board.drop_token(move)
                score = self._minimax(new_board, depth - 1, False)
                max_score = max(max_score, score)
            return max_score
        else:
            min_score = float('inf')
            for move in board.possible_drops():
                new_board = deepcopy(board)
                new_board.drop_token(move)
                score = self._minimax(new_board, depth - 1, True)
                min_score = min(min_score, score)
            return min_score

    def _score(self, board):
        counter = 0
        if board.board[0][0] != self.my_token and board.board[0][0] != '_':
            counter += 0.2
        if board.board[board.height - 1][board.width - 1] != self.my_token and board.board[board.height - 1][board.width - 1] != '_':
            counter += 0.2
        if board.board[board.height - 1][0] != self.my_token and board.board[board.height - 1][0] != '_':
            counter += 0.2
        if board.board[0][board.width - 1] != self.my_token and board.board[0][board.width - 1] != '_':
            counter += 0.2
        #print("------", counter)
        return counter
