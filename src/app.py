import os
import random
import time # type: ignore
from typing import List
from game_state import GameState
from move_handlers.mcts_handler import MCTSHandler # type: ignore
from move_handlers.console_handler import ConsoleHandler  # type: ignore
from move_handlers.minimax_handler import MinimaxHandler  # type: ignore
from move_handlers.minimax_iterative_handler import MinimaxIterativeHandler  # type: ignore
from move_handlers.move_handler import MoveHandler
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

    file_name = "boards_dataset.csv"
    


    #with open("model_comparison.csv", "a") as f:
    #    f.write(f"Model 1;Model 2;Wins X;Wins Y;Draws\n")

    # Random vs Minimax
    #compare_models(RandomHandler("X", False), MinimaxIterativeHandler("O", 0.1, False), 1000)
    #compare_models(MinimaxIterativeHandler("X", 0.1, False), RandomHandler("O", False), 1000)
    #
    ## Random vs MCTS
    #compare_models(RandomHandler("X", False), MCTSHandler("O", 0.1, False), 1000)
    #compare_models(MCTSHandler("X", 0.1, False), RandomHandler("O", False), 1000)

    # MCTS vs Minimax
    # compare_models(MCTSHandler("X", 0.1, False), MinimaxIterativeHandler("O", 0.1, False), 1000)
    # compare_models(MinimaxIterativeHandler("X", 0.1, False), MCTSHandler("O", 0.1, False), 1000)
    # print("All comparisons done")

    # MCTS moves with increasing time limits
    # compare_runtime([5])

    # compare_models(MinimaxIterativeHandler("X", 0.1, False), MinimaxIterativeHandler("O", 0.1, False), 1000)

    # analyse_branching_factor()

    print("All done")

def create_dataset(board_cound: int, file_name: str):
    if not os.path.isfile(file_name):
        with open(file_name, "w") as f:
            header = f"i;{';'.join(f'cell{i}' for i in range(81))};forced_board;winner\n"
            f.write(header)

    progress_bar = tqdm(
        range(board_cound),
        desc="Creating dataset",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    )

    for i in progress_bar:
        player_dumb = RandomHandler("X", False)
        player_smart = MinimaxHandler("O", False)
        last_move_naive = random.randint(0, 70)
        
        state = GameState(player_dumb, player_smart, last_move_naive, index=i, file_name=file_name)
        state.run_game_praktikum() # automatically creates the entries for one board


def compare_runtime(time_limits: List[float]):
    # with open("runtime_comparison.csv", "w") as f:
    #     f.write(f"Model 1;Model 2;Runtime 1; Runtime 2;Wins X;Wins Y;Draws;Max-Value 1;Max-Value 2\n")
        
    for i in time_limits:
        model1 = MCTSHandler
        model2 = MinimaxIterativeHandler
        run_comparison(model1, model2, i) # increase time limit for MCTS
        run_comparison(model2, model1, i) # increase time limit for Minimax
            
def run_comparison(model1: type[MCTSHandler] | type[MinimaxIterativeHandler], model2: type[MCTSHandler] | type[MinimaxIterativeHandler], time: float):
    x_wins = 0
    o_wins = 0
    draws = 0
    progress_bar = tqdm(
        range(50),
        desc=f"{model1.__name__}: {time} seconds vs {model2.__name__}: 0.1 seconds",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    )
    algo1 = None
    algo2 = None
    for _ in progress_bar:
        try:
            algo1 = model1("X", time, False)
            algo2 = model2("O", 0.1, False)
            board = UTTTBoard()
            state = GameState(board, algo1, algo2)
            state.run_game_loop(False)

            if board.winner == "X": 
                x_wins += 1
            elif board.winner == "O":
                o_wins += 1
            else:
                draws += 1
        except KeyboardInterrupt:
            print("\nGame interrupted. Exiting...")
            return
        
    with open("runtime_comparison.csv", "a") as f:
        f.write(f"{model1.__name__};{model2.__name__};{time};0.1;{x_wins};{o_wins};{draws};{algo1.get_max_value() if algo1 else 0};{algo2.get_max_value() if algo2 else 0}\n")

def compare_models(playerX: MoveHandler, playerO: MoveHandler, games: int = 100):
    x_wins = 0
    o_wins = 0
    draws = 0

    progress_bar = tqdm(
        range(games),
        desc=f"{playerX.__class__.__name__} vs {playerO.__class__.__name__}",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    )

    for _ in progress_bar:
        board = UTTTBoard()
        state = GameState(board, playerX, playerO)

        try:
            state.run_game_loop(False)
        except KeyboardInterrupt:
            print("\nGame interrupted. Exiting...")
            break

        if board.winner == "X":
            x_wins += 1
        elif board.winner == "O":
            o_wins += 1
        else:
            draws += 1

    with open("model_comparison.csv", "a") as f:
        f.write(f"{playerX.__name__};{playerO.__name__};{x_wins};{o_wins};{draws}\n")
    
def analyse_branching_factor():
    with open("branching_factor.csv", "w") as f:
        f.write("Move;Possible Moves Total;Simulation Count\n")

    branching_coll = dict[int, tuple[int, int]]()

    progress_bar = tqdm(
        range(300),
        desc="Branching Factor Analysis",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
    )
    
    for _ in progress_bar:
        handler1 = MCTSHandler("X", 0.1, False)
        handler2 = MCTSHandler("O", 0.1, False)
        board = UTTTBoard()
        state = GameState(board, handler1, handler2)
        state.run_game_loop(False)

        for move, count in handler1.branching_factor.items():
            if move not in branching_coll:
                branching_coll[move] = count, 1
            else:
                branching_coll[move] = count + branching_coll[move][0], 1 + branching_coll[move][1]
        
        for move, count in handler2.branching_factor.items():
            if move not in branching_coll:
                branching_coll[move] = count, 1
            else:
                branching_coll[move] = count + branching_coll[move][0], 1 + branching_coll[move][1]

    with open("branching_factor.csv", "a") as f:
        for move, count in branching_coll.items():
            f.write(f"{move};{count[0]};{count[1]}\n")

if __name__ == "__main__":
    main()
