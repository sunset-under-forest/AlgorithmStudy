# connect four game
import platform
from os import system

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

    def play_game(self):
        game_over = False

        while not game_over:
            # Player 1 input
            if self.player == 1:
                col = int(input("Player 1 (O) make your selection (0-6): "))
            # Player 2 input
            else:
                col = int(input("Player 2 (X) make your selection (0-6): "))

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
