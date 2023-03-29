# 皇后问题递归解决
import numpy as np

ways_nums = 0


def show_chess_board(chessBoard):
    """
    information: 打印期盼
    creating time: 2023/3/17
    """
    size = chessBoardSize
    print("\t", end="")
    for i in range(size):
        print(i, end='\t')
    print()
    for row in range(size):
        print(row, end="\t")
        for col in range(size):
            print("■" if chessBoard[row][col] == 1 else "□", end="\t")
        print()


def settle_queens(queens):
    chessBoard = np.zeros((chessBoardSize, chessBoardSize), dtype="byte")
    for row, col in enumerate(queens):
        # col == queens[row]
        chessBoard[row][col] = 1
    return chessBoard


def is_queen_not_conflict(queens, row):
    for queen_idx in range(row):
        for another_queen_idx in range(queen_idx + 1, row):
            if queens[queen_idx] == queens[another_queen_idx]:  # 列相同与否
                return False

            row_distance = another_queen_idx - queen_idx  # 一定大于零
            col_distance = abs(queens[another_queen_idx] - queens[queen_idx])
            if row_distance == col_distance:  # 是否处于同一对角线
                return False
    return True


# print(is_queen_not_conflict(queens, 1))

iter_steps = 0


def solution(queens, row=0):
    global ways_nums, iter_steps
    for col in range(chessBoardSize):
        iter_steps += 1
        queens[row] = col  # (row,col)
        # print(iter_steps, queens)
        if is_queen_not_conflict(queens, row + 1):
            if row == chessBoardSize - 1:
                ways_nums += 1
                print(f"way {ways_nums}:")
                show_chess_board(settle_queens(queens))
                return
            solution(queens, row + 1)
    return

    ...

    return chessBoard


chessBoardSize = 8
queens = np.zeros(chessBoardSize, dtype="byte")
solution(queens)
