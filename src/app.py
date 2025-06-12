import os
import random
import time # type: ignore
from typing import List
from game_state import GameState
from move_handlers.mcts_handler import MCTSHandler # type: ignore
from move_handlers.console_handler import ConsoleHandler  # type: ignore
from move_handlers.minimax_handler import MinimaxHandler  # type: ignore
from move_handlers.minimax_iterative_handler import MinimaxIterativeHandler  # type: ignore
from move_handlers.random_handler import RandomHandler  # type: ignore
from uttt_board import BoardIndex, UTTTBoard
from tqdm import tqdm

def modify_board(board: UTTTBoard) -> None:
    """Sets the board state to match the provided ASCII scenario."""

    x_won_boards: List[BoardIndex] = [2, 4, 8]
    o_won_boards: List[BoardIndex] = [0, 1, 5, 6]

    for idx in x_won_boards:
        board.get_small_board(idx).winner = "X"
    for idx in o_won_boards:
        board.get_small_board(idx).winner = "O"

    board.get_small_board(3).winner = "Draw"

    board2 = board.get_small_board(7)
    board2.make_move(0, "X")
    board2.make_move(1, "X")
    board2.make_move(2, "O")
    board2.make_move(3, "O")
    board2.make_move(4, "O")
    board2.make_move(5, "X")
    board2.make_move(7, "O")
    board2.make_move(8, "X")

def main():
    print("Running...")

    create_dataset(10_000, "boards_dataset.csv")

    print("All done")

def create_dataset(board_cound: int, file_name: str):
    if not os.path.isfile(file_name):
        with open(file_name, "w") as f:
            header = f"i;{';'.join(f'cell{i}' for i in range(81))};forced_board;winner;confidence\n"
            f.write(header)

    progress_bar = tqdm(
        range(board_cound),
        desc=f"Creating dataset in file {file_name}",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    )

    for i in progress_bar:
        player_dumb = RandomHandler("X", False)
        player_smart = MinimaxHandler("O", False)
        last_move_naive = random.randint(0, 70)
        
        state = GameState(player_dumb, player_smart, last_move_naive, index=i, file_name=file_name)
        state.run_game_praktikum() # automatically creates the entries for one board