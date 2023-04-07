# connect four game
import platform
import random
from os import system
from math import inf as infinity

import numpy as np


class ConnectFour:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols

        self.board = np.zeros((rows, cols))
        self.player = 1

    def drop_piece(self, col):
        if self.is_valid_location(col):
            row = self.get_next_open_row(col)
            self.board[row][col] = self.player
            return True
        else:
            return False

    def clean(self):
        """
        Clears the console
        """
        os_name = platform.system().lower()
        if 'windows' in os_name:
            system('cls')
        else:
            system('clear')

    def is_valid_location(self, col):
        return self.board[self.rows - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(6):
            if self.board[r][col] == 0:
                return r

    def render(self, state, c_choice, h_choice):
        """
        Print the board on console
        :param state: current state of the board
        """

        chars = {
            -1: h_choice,
            +1: c_choice,
            0: ' '
        }
        str_line = '-----' * self.cols

        print('\n' + str_line)
        for row in state:
            for cell in row:
                symbol = chars[cell]
                print(f'| {symbol} |', end='')
            print('\n' + str_line)

    def print_board(self):
        self.clean()
        self.render(np.flip(self.board, 0), 'O', 'X')

    def winning_move(self, piece):
        # Check horizontal locations
        for c in range(self.cols - 3):
            for r in range(self.rows):
                if self.board[r][c] == piece and self.board[r][c + 1] == piece and self.board[r][c + 2] == piece and \
                        self.board[r][c + 3] == piece:
                    return True

        # Check vertical locations
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if self.board[r][c] == piece and self.board[r + 1][c] == piece and self.board[r + 2][c] == piece and \
                        self.board[r + 3][c] == piece:
                    return True

        # Check diagonal 1 locations
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and self.board[r + 2][
                    c + 2] == piece and self.board[r + 3][c + 3] == piece:
                    return True

        # Check diagonal 2 locations
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if self.board[r][c] == piece and self.board[r - 1][c + 1] == piece and self.board[r - 2][
                    c + 2] == piece and self.board[r - 3][c + 3] == piece:
                    return True

    def switch_player(self):
        if self.player == 1:
            self.player = -1
        else:
            self.player = 1

    def get_valid_locations(self):
        valid_locations = []
        for col in range(self.cols):
            if self.is_valid_location(col):
                valid_locations.append(col)
        return valid_locations

    def get_current_score(self):
        score = 0
        center_array = [int(i) for i in list(self.board[:, self.cols // 2])]  # center column
        center_count = center_array.count(self.player)
        score += center_count * 3  # center column is more important

        # horizontal
        for r in range(self.rows):
            row_array = [int(i) for i in list(self.board[r, :])]  # go through each row
            for c in range(self.cols - 3):
                window = row_array[c:c + 4]
                score += self.evaluate_window(window)

        # vertical
        for c in range(self.cols):
            col_array = [int(i) for i in list(self.board[:, c])]  # go through each column
            for r in range(self.rows - 3):
                window = col_array[r:r + 4]
                score += self.evaluate_window(window)

        # positive sloped diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [self.board[r + i][c + i] for i in range(4)]
                # go through each diagonal from top left to bottom right
                score += self.evaluate_window(window)

        # negative sloped diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [self.board[r + 3 - i][c + i] for i in range(4)]
                # go through each diagonal from bottom left to top right
                score += self.evaluate_window(window)

        return score

    def evaluate_window(self, window):
        score = 0
        opp_piece = self.player * -1

        if window.count(self.player) == 4:
            score += 100
        elif window.count(self.player) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(self.player) == 2 and window.count(0) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 4

        return score

    def is_game_over(self):
        return self.winning_move(self.player) or len(self.get_valid_locations()) == 0

    # minimax algorithm
    def min_max(self, depth, maximizingPlayer=1, alpha=-infinity, beta=infinity):
        valid_locations = self.get_valid_locations()
        is_terminal = self.is_game_over()
        if depth == 0 or is_terminal:
            return [None, self.get_current_score()]

        best = [None, -infinity] if maximizingPlayer else [None, infinity]

        for col in valid_locations:
            row = self.get_next_open_row(col)
            self.drop_piece(col)
            score = self.min_max(depth - 1, -1 * maximizingPlayer, alpha, beta)[1]
            self.board[row][col] = 0
            if maximizingPlayer:
                if score > best[1]:
                    best = [col, score]
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            else:
                if score < best[1]:
                    best = [col, score]
                beta = min(beta, score)
                if alpha >= beta:
                    break

        return best

    def play_game(self):
        game_over = False
        while not game_over:

            # col, minimax_score = self.min_max(4, self.player)
            # # show the minimax score and player
            # print("Player", "O" if self.player == 1 else "X", "minimax score:", minimax_score)
            if self.player == 1:
                col = int(input("Player 1 (O) make your selection (0-" + str(self.cols - 1) + "):"))
            else:
                col, minimax_score = self.min_max(7, self.player)
                print("Player 2 (X) make your selection (0-" + str(self.cols - 1) + "):", col)


            if self.drop_piece(col):
                if self.winning_move(self.player):
                    self.print_board()
                    if self.player == 1:
                        print("Player 1 (O) wins!")
                    else:
                        print("Player 2 (X) wins!")
                    game_over = True
                else:
                    self.switch_player()
                    self.print_board()


if __name__ == '__main__':
    game = ConnectFour(4, 5)
    game.play_game()
