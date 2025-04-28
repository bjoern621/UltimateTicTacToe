import pytest
from typing import List, Tuple

from ttt_board import Player, TTTBoard


# Helper function to create a board from a list of moves
def create_board_from_moves(
    moves: List[Tuple[int, Player]],
) -> TTTBoard:
    board = TTTBoard()
    for index, player in moves:
        # Accessing private attribute intentionally for testing internal logic
        board._TTTBoard__board[index] = player  # type: ignore
    return board


# Test cases for X wins
@pytest.mark.parametrize(
    "moves",
    [
        # Horizontal wins
        [(0, "X"), (1, "X"), (2, "X")],
        [(3, "X"), (4, "X"), (5, "X")],
        [(6, "X"), (7, "X"), (8, "X")],
        # Vertical wins
        [(0, "X"), (3, "X"), (6, "X")],
        [(1, "X"), (4, "X"), (7, "X")],
        [(2, "X"), (5, "X"), (8, "X")],
        # Diagonal wins
        [(0, "X"), (4, "X"), (8, "X")],
        [(2, "X"), (4, "X"), (6, "X")],
    ],
)
def test_check_winner_x_wins(moves: List[Tuple[int, Player]]):  # Added type hint
    board = create_board_from_moves(moves)
    assert board._TTTBoard__check_winner() == "X"  # type: ignore


# Test cases for O wins
@pytest.mark.parametrize(
    "moves",
    [
        # Horizontal wins
        [(0, "O"), (1, "O"), (2, "O")],
        [(3, "O"), (4, "O"), (5, "O")],
        [(6, "O"), (7, "O"), (8, "O")],
        # Vertical wins
        [(0, "O"), (3, "O"), (6, "O")],
        [(1, "O"), (4, "O"), (7, "O")],
        [(2, "O"), (5, "O"), (8, "O")],
        # Diagonal wins
        [(0, "O"), (4, "O"), (8, "O")],
        [(2, "O"), (4, "O"), (6, "O")],
    ],
)
def test_check_winner_o_wins(moves: List[Tuple[int, Player]]):  # Added type hint
    board = create_board_from_moves(moves)
    assert board._TTTBoard__check_winner() == "O"  # type: ignore


# Test case for a draw
def test_check_winner_draw():
    moves: List[Tuple[int, Player]] = [
        (0, "X"),
        (1, "O"),
        (2, "X"),
        (3, "X"),
        (4, "O"),
        (5, "O"),
        (6, "O"),
        (7, "X"),
        (8, "X"),
    ]
    board = create_board_from_moves(moves)
    assert board._TTTBoard__check_winner() == "Draw"  # type: ignore


# Test cases for no winner (ongoing game)
@pytest.mark.parametrize(
    "moves",
    [
        [],  # Empty board
        [(0, "X")],
        [(0, "X"), (1, "O")],
        [(0, "X"), (4, "O"), (8, "X")],
        [
            (0, "X"),
            (1, "O"),
            (2, "X"),
            (3, "X"),
            (4, "O"),
            (5, "O"),
            (6, "O"),
            (7, "X"),  # (8 is None)
        ],
    ],
)
def test_check_winner_no_winner(moves: List[Tuple[int, Player]]):
    board = create_board_from_moves(moves)
    assert board._TTTBoard__check_winner() is None  # type: ignore
