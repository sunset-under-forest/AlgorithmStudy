# connect four game
import platform
import random
from os import system
from math import inf as infinity

import numpy as np

MAX = +1
MIN = -1


class ConnectFour:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols

        self.board = np.zeros((rows, cols))
        self.player = 1

    def drop_piece(self, col , player=0):

        if self.is_valid_location(col):
            row = self.get_next_open_row(col)
            self.board[row][col] = self.player if player == 0 else player
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

    def chessvalue(self, l, d1, d2):

        if l < 4 and d1 == d2 == 0: return 0
        value = 10 ** (l + 2)
        k1 = d1 * d2
        k2 = d1 + d2

        v1 = 0 if d1 == 0 else 10 ** (l + 1) / d1 ** 3
        v2 = 0 if d2 == 0 else 10 ** (l + 1) / d2 ** 3
        value += v1 + v2
        if k1 > 0:
            value += (10 ** (l + 1) / k1 ** 1.5)
        if k2 > 0:
            value += (10 ** (l) / k2)
        # value *= 100
        return int(value)

    def all_chess(self, board=None):
        """
        calculate the value of all chess
        气度：0代表这个方向上没有可以使串延伸的空位，n(n>0)代表还需要n步才可以在这个方向上的空位上落子，使串延伸
        """
        # 判断子串重复
        check_repeat = lambda m, n: all(map(lambda x: x in n, m))  # 从m中找出n中的元素

        ret = {1: [], -1: []}  # 1:O, -1:X
        # 每个点沿右，右下，下，左下4个方向最多走3步，每一步都必须与当前点同色，如果能走到3步则表示获胜
        directions = [[0, 1], [1, 1], [1, 0], [1, -1]]  # vector of direction
        board = self.board if board is None else board
        for x in range(self.rows):
            for y in range(self.cols):
                position = board[x, y]
                if position != 0:  # O
                    # 从当前点开始，沿4个方向走
                    for direction in directions:
                        step = 0
                        chess = [(x, y)]  # 当前串
                        xn, yn = x, y  # 下一个点的位置
                        # ----- 计算气度1:d1
                        d1 = 0  # 气度1的延申方向和当前方向相反
                        px, py = x - direction[0], y - direction[1]
                        if 0 <= px < self.rows and 0 <= py < self.cols:  # 判断点是否在区域内，如果在区域内则计算气度，否则气度为0，代表这个方向上没有可以使串延伸的空位
                            if board[px, py] == 0:  # 如果是空位
                                k = board[:, py][::-1][:self.rows - px]  # 取出竖直方向上的空位
                                d1 = len(list(filter(lambda x: x == 0, k)))  # 计算空位的个数，若无空位则气度为0，否则气度为空位的个数
                        # -----
                        # 最大深入3级
                        for s in range(3):
                            # 计算下一个点的位置
                            xn, yn = xn + direction[0], yn + direction[1]  # 下一个点的位置和当前方向相同
                            # 判断点是否在区域内，这一部分是为了找出当前串的长度
                            if 0 <= xn < self.rows and 0 <= yn < self.cols:
                                if board[xn, yn] == position:
                                    chess.append((xn, yn))
                                    step += 1
                                else:
                                    break
                            else:
                                break
                        # ----- 计算气度2：d2
                        d2 = 0  # 气度2的延申方向和当前方向相同
                        # xn,yn = xn+d[0],yn+d[1]
                        # 此时xn,yn已经是下一个点的位置，所以不需要再加上方向
                        if 0 <= xn < self.rows and 0 <= yn < self.cols:
                            if board[xn, yn] == 0:  # 如果是空位
                                k = board[:, yn][::-1][:self.rows - xn]  # 取出竖直方向上的空位
                                d2 = len(list(filter(lambda x: x == 0, k)))  # 计算空位的个数，若无空位则气度为0，否则气度为空位的个数
                        # -----
                        # 记录棋串
                        # 判断是否重复
                        if not any(map(lambda x: check_repeat(chess, x[0]), ret[position])):
                            value = self.chessvalue(len(chess), d1, d2)
                            ret[position].append([chess, [len(chess), d1, d2], position*value])
                        # 记录棋串长度，气度1，气度2

                        # # 不判断重复
                        # value = self.chessvalue(len(chess), d1, d2)
                        # ret[position].append([chess, [len(chess), d1, d2], position, position * value])

        # 棋串排序,按棋长
        # mysort = lambda r: sorted(r,key=lambda x:-len(x[0]))
        # 棋串排序,按棋串得分
        mysort = lambda r: sorted(r, key=lambda x: -x[-1])
        for k, v in ret.items():
            ret[k] = mysort(v)
        return ret

    def get_current_score_old(self):
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

    def get_current_score(self, player):
        """
        get the current score by all_chess
        :param player: 1 or -1
        :return: score
        """
        score = 0
        for chess in self.all_chess(np.flip(self.board, 0))[player]:
            # print(chess)
            score += chess[-1]
        for chess in self.all_chess(np.flip(self.board, 0))[-player]:
            # print(chess)
            score += chess[-1]
        # print(score,player)
        return score

    def is_game_over(self):
        return self.winning_move(self.player) or len(self.get_valid_locations()) == 0

    # minimax algorithm
    def min_max(self, depth, maximizingPlayer=MAX, alpha=-infinity, beta=infinity):
        valid_locations = self.get_valid_locations()
        is_terminal = self.is_game_over()
        if depth == 0 or is_terminal:
            score = self.get_current_score(maximizingPlayer)
            # print("depth:", depth , "score:", score)
            # self.print_board()
            return [-1, score]

        best = [-1, -infinity] if maximizingPlayer == MAX else [-1, infinity]
        # print("best:", best)
        for col in valid_locations:
            row = self.get_next_open_row(col)
            self.drop_piece(col, maximizingPlayer)
            score = self.min_max(depth - 1, -1 * maximizingPlayer, alpha, beta)[1]
            self.board[row][col] = 0
            if maximizingPlayer == MAX:
                if score > best[1]:
                    best = [col, score]
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            elif maximizingPlayer == MIN:
                if score < best[1]:
                    best = [col, score]
                beta = min(beta, score)
                if alpha >= beta:
                    break

        return best

    def play_game(self):
        game_over = False
        self.switch_player()
        while not game_over:
            # try:
            #     if minimax_score == -205720.0:
            #         print("stop!")
            # except:
            #     ...
            # col, minimax_score = self.min_max(7, self.player)
            # show the minimax score and player
            # print("Player", "O" if self.player == 1 else "X", "minimax score:", minimax_score)
            # if minimax_score == -210900.0:
            #     print("stop!")
            if self.player == 1:
                col = int(input("Player 1 (O) make your selection (0-" + str(self.cols - 1) + "):"))
            else:
                col, minimax_score = self.min_max(4, self.player)
                print("Player 2 (X) make your selection (0-" + str(self.cols - 1) + "):", col)
                print("Player 2 (X) minimax score:", minimax_score)

            if self.drop_piece(col):
                if self.winning_move(self.player):
                    self.print_board()
                    if self.player == 1:
                        print("Player 1 (O) wins!")
                    else:
                        print("Player 2 (X) wins!")
                    game_over = True
                elif len(self.get_valid_locations()) == 0:
                    self.print_board()
                    print("Game over. Nobody wins!")
                    game_over = True
                else:
                    self.switch_player()
                    self.print_board()


if __name__ == '__main__':
    game = ConnectFour(4, 5)
    game.play_game()
