from typing import cast

from move_handlers.minimax_iterative_handler import MinimaxIterativeHandler
from ttt_board import Winner
from uttt_board import BoardStateHash, UTTTBoard, BoardIndex, CellIndex


def set_board_state(board: UTTTBoard, state: BoardStateHash):
    """Sets the winner of each small board for testing purposes."""

    assert len(state) == 9, "Test state tuple must have exactly 9 elements."

    for i, small_board_state in enumerate(state):
        board_index = cast(BoardIndex, i)
        small_board = board.get_small_board(board_index)

        if small_board_state in [None, "X", "O"]:
            small_board_state = cast(Winner, small_board_state)
            small_board.winner = small_board_state
        else:
            assert isinstance(
                small_board_state, tuple
            ), "Expected a tuple for cell values"

            for j, cell_value in enumerate(small_board_state):
                cell_index = cast(CellIndex, j)

                if cell_value in ["X", "O"]:
                    small_board.make_move(cell_index, cell_value)


# def test_iterative_deepening_finds_move_after_draw():
#     board = UTTTBoard()
#     state: BoardStateHash = (
#         "O",
#         ("X", None, "X", None, None, None, None, "O", "O"),
#         "X",
#         "X",
#         "O",
#         "O",
#         ("O", "X", "O", "O", "X", "X", "X", None, "O"),
#         "O",
#         ("O", "X", "O", "X", None, None, "X", "X", "O"),
#     )
#     __set_board_state(board, state)

#     board.display_board(4)

#     # # Player X to move, forced into board 8 (index 8)
#     # forced_board = cast(BoardIndex, 8)

#     move_handler = MinimaxIterativeHandler("X", 3)

#     move_handler.get_move(board, None)


def test_stop_all_children_terminal():
    board = UTTTBoard()
    state: BoardStateHash = (
        "X",
        (None, "O", None, "X", None, None, "X"),
        "O",
        "O",
        "O",
        "X",
        "X",
        ("O", None, "X", None, None, "X", None, "O", "O"),
        "X",
    )
    set_board_state(board, state)

    board.display_board(4)

    move_handler = MinimaxIterativeHandler("X", 3)

    move_handler.get_move(board, None)

    # There are 10 possible moves. The game ends in a draw. Run pytest -s to see the output.
    # At depth 10, the function should stop searching all children and return the best move.
