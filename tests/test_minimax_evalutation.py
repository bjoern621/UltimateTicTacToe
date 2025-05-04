from move_handlers.minimax_handler import MinimaxHandler
from tests.test_iterative_depth import set_board_state
from uttt_board import BoardStateHash, UTTTBoard


def test_stop_all_children_terminal():
    board = UTTTBoard()
    state: BoardStateHash = (
        "O",
        None,
        "O",
        None,
        "X",
        ("X", "X", "O", "X", None, None, "O", None, None),
        None,
        "X",
        "X",
    )
    set_board_state(board, state)

    board.display_board(None)

    move_handler = MinimaxHandler("X", 0)

    score: float = move_handler._MinimaxHandler__evaluate_board(board)  # type: ignore

    print(f"Score: {score}")

    assert score == 159.0, "Score should be 159.0"
