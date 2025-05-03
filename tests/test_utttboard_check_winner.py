import pytest
from typing import List, Tuple, Optional

from ttt_board import Winner
from uttt_board import UTTTBoard


# Helper function to create a UTTTBoard with predefined sub-board winners
def create_uttt_board_with_winners(
    winners: List[Tuple[int, Optional[Winner]]],
) -> UTTTBoard:
    """Creates a UTTTBoard and sets the winners of its sub-boards."""

    uttt_board = UTTTBoard()

    for index, winner in winners:
        # Accessing private attribute intentionally for testing internal logic
        # We assume the sub-board's internal state doesn't matter for this test,
        # only its reported winner.
        uttt_board._UTTTBoard__small_boards[index].winner = winner  # type: ignore

    return uttt_board


# Test cases for X winning the overall game
@pytest.mark.parametrize(
    "winners",
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
def test_uttt_check_winner_x_wins(winners: List[Tuple[int, Optional[Winner]]]):
    """Tests if UTTTBoard correctly identifies X as the overall winner."""

    board = create_uttt_board_with_winners(winners)
    assert board._UTTTBoard__check_winner() == "X"  # type: ignore


# Test cases for O winning the overall game
@pytest.mark.parametrize(
    "winners",
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
def test_uttt_check_winner_o_wins(winners: List[Tuple[int, Optional[Winner]]]):
    """Tests if UTTTBoard correctly identifies O as the overall winner."""

    board = create_uttt_board_with_winners(winners)
    assert board._UTTTBoard__check_winner() == "O"  # type: ignore


# Test case for an overall draw
def test_uttt_check_winner_draw():
    """Tests if UTTTBoard correctly identifies an overall draw."""

    # All boards are decided (won or drawn), but no line of 3 winners
    winners: List[Tuple[int, Optional[Winner]]] = [
        (0, "X"),
        (1, "O"),
        (2, "X"),
        (3, "O"),
        (4, "X"),
        (5, "O"),
        (6, "O"),
        (7, "X"),
        (8, "Draw"),
    ]

    board = create_uttt_board_with_winners(winners)
    assert board._UTTTBoard__check_winner() == "Draw"  # type: ignore


# Test cases for no overall winner yet (ongoing game)
@pytest.mark.parametrize(
    "winners",
    [
        [],  # Empty board (all sub-boards ongoing)
        [(0, "X")],
        [(0, "X"), (1, "O")],
        [(0, "X"), (4, "O"), (8, "X")],  # Diagonal, but not a winning line yet
        [  # A full board, but one square is still playable (None)
            (0, "X"),
            (1, "O"),
            (2, "X"),
            (3, "O"),
            (4, "X"),
            (5, "O"),
            (6, "O"),
            (7, "X"),  # (8 is None)
        ],
    ],
)
def test_uttt_check_winner_no_winner(winners: List[Tuple[int, Optional[Winner]]]):
    """Tests scenarios where the overall game has not concluded."""

    board = create_uttt_board_with_winners(winners)
    assert board._UTTTBoard__check_winner() is None  # type: ignore
