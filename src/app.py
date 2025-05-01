import time
from typing import List
from game_state import GameState
from move_handlers.console_handler import ConsoleHandler  # type: ignore
from move_handlers.minimax_handler import MinimaxHandler  # type: ignore
from move_handlers.random_handler import RandomHandler  # type: ignore
from uttt_board import BoardIndex, UTTTBoard


def modify_board(board: UTTTBoard) -> None:
    """Sets the board state to match the provided ASCII scenario."""

    # --- Set winners for completed boards ---
    x_won_boards: List[BoardIndex] = [2, 4, 5, 7]
    o_won_boards: List[BoardIndex] = [0, 1, 6, 8]

    for idx in x_won_boards:
        board.get_small_board(idx).winner = "X"
    for idx in o_won_boards:
        board.get_small_board(idx).winner = "O"

    board3 = board.get_small_board(3)
    board3.make_move(0, "O")
    board3.make_move(2, "O")
    board3.make_move(7, "O")
    board3.make_move(4, "X")
    board3.make_move(8, "X")


def main():
    print("Running...")

    x_wins = 0
    o_wins = 0
    draws = 0

    games = 1

    start_time = time.time()

    for i in range(games):
        board = UTTTBoard()
        # modify_board(board)

        # playerX = ConsoleHandler("X")
        # playerO = ConsoleHandler("O")

        # playerO = RandomHandler("O")2

        playerX = MinimaxHandler("X", 5)

        state = GameState(board, playerX, playerO)

        state.run_game_loop()

        print(playerX.overall_runtime)

        if board.winner == "X":
            x_wins += 1
        elif board.winner == "O":
            o_wins += 1
        else:
            draws += 1

        print(f"Finished game {i + 1} of {games}")

    print(f"X wins: {x_wins}")
    print(f"O wins: {o_wins}")
    print(f"Draws: {draws}")

    print(f"Time taken: {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    main()
