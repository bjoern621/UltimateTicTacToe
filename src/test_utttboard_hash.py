from typing import Dict, Tuple
import pytest
from uttt_board import UTTTBoard, BoardStateHash


def test_hash_consistency():
    """Tests that the same board state produces the same hash."""

    board1 = UTTTBoard()
    board1.make_move(0, 0, "X")
    board1.make_move(1, 1, "O")
    board1.make_move(4, 4, "X")

    hash1 = board1.get_hashable_state()

    # Create an identical board state through copying
    board2 = board1.copy()
    hash2 = board2.get_hashable_state()

    # Create an identical board state manually
    board3 = UTTTBoard()
    board3.make_move(0, 0, "X")
    board3.make_move(1, 1, "O")
    board3.make_move(4, 4, "X")
    hash3 = board3.get_hashable_state()

    assert hash1 == hash2, "Hash from copied board should be the same"
    assert (
        hash1 == hash3
    ), "Hash from manually created identical board should be the same"
    # Check Python's built-in hash function as well
    assert hash(hash1) == hash(
        hash2
    ), "Python's hash() should be consistent for equal states"
    assert hash(hash1) == hash(
        hash3
    ), "Python's hash() should be consistent for equal states"


def test_hash_difference():
    """Tests that different board states produce different hashes."""

    board1 = UTTTBoard()
    board1.make_move(0, 0, "X")
    hash1 = board1.get_hashable_state()

    board2 = UTTTBoard()
    board2.make_move(0, 1, "X")  # Different move
    hash2 = board2.get_hashable_state()

    board3 = UTTTBoard()
    board3.make_move(0, 0, "O")  # Different player
    hash3 = board3.get_hashable_state()

    board4 = UTTTBoard()
    board4.make_move(1, 0, "X")  # Different board index
    hash4 = board4.get_hashable_state()

    # Test small board winner difference
    board5 = UTTTBoard()
    board5.make_move(0, 0, "X")
    board5.make_move(0, 1, "X")
    board5.make_move(0, 2, "X")  # Board 0 won by X
    hash5 = board5.get_hashable_state()

    board6 = UTTTBoard()  # Empty board
    hash6 = board6.get_hashable_state()

    assert hash1 != hash2, "Different moves should produce different hashes"
    assert hash1 != hash3, "Different players should produce different hashes"
    assert hash1 != hash4, "Moves on different boards should produce different hashes"
    assert (
        hash1 != hash5
    ), "Different small board winners should produce different hashes"
    assert hash1 != hash6, "Non-empty vs empty board should produce different hashes"


def test_hash_usability_as_dict_key():
    """Tests that the returned hash can be used as a dictionary key."""

    board1 = UTTTBoard()
    board1.make_move(3, 5, "O")
    hash1 = board1.get_hashable_state()

    board2 = UTTTBoard()
    board2.make_move(3, 6, "X")
    hash2 = board2.get_hashable_state()

    memo: Dict[
        BoardStateHash,
        Tuple[str, Tuple[int, int]],
    ] = {}
    try:
        memo[hash1] = ("value1", (3, 5))
        memo[hash2] = ("value2", (3, 6))
    except TypeError:
        pytest.fail(
            "BoardStateHash could not be used as a dictionary key (not hashable)"
        )

    assert memo[hash1] == ("value1", (3, 5))
    assert memo[hash2] == ("value2", (3, 6))
    assert len(memo) == 2
    assert hash1 in memo
    assert hash2 in memo


def test_hash_includes_small_board_winners():
    """Tests that the hash changes when a small board winner changes, even if cells are the same."""

    board1 = UTTTBoard()
    board1.make_move(0, 0, "X")
    board1.make_move(0, 1, "X")
    board1.make_move(0, 2, "X")  # Board 0 won by X
    hash1 = board1.get_hashable_state()

    board2 = UTTTBoard()
    board2.make_move(0, 0, "X")
    board2.make_move(0, 1, "X")
    board2.make_move(0, 2, "X")
    # Manually set winner to O to create an inconsistent state for testing hash
    # This simulates a scenario where the hash must rely on the stored winner, not just cell values
    board2.get_small_board(0).winner = "O"
    hash2 = board2.get_hashable_state()

    assert hash1 != hash2, "Hash should differ if small board winner differs"


def test_hash_same_state_different_move_order():
    """Tests that the hash remains the same even if the cells are in different orders."""

    board1 = UTTTBoard()
    board1.make_move(0, 0, "X")
    board1.make_move(0, 1, "X")
    board1.make_move(0, 2, "X")  # Board 0 won by X
    hash1 = board1.get_hashable_state()

    board2 = UTTTBoard()
    board2.make_move(0, 2, "X")
    board2.make_move(0, 1, "X")
    board2.make_move(0, 0, "X")  # Same moves in different order
    hash2 = board2.get_hashable_state()

    assert (
        hash1 == hash2
    ), "Hash should be the same for identical states in different orders"


def test_hash_empty_board():
    """Tests that the hash of an empty board is consistent."""

    board1 = UTTTBoard()
    hash1 = board1.get_hashable_state()

    board2 = UTTTBoard()
    hash2 = board2.get_hashable_state()

    assert hash1 == hash2, "Hash of empty boards should be the same"


def test_hash_cells_irrelevant_if_winner_is_same():
    """Tests that the hash remains the same if the winner is the same, regardless of cell values."""

    board1 = UTTTBoard()
    board1.make_move(0, 0, "X")
    board1.make_move(0, 1, "X")
    board1.make_move(0, 2, "X")  # Board 0 won by X
    hash1 = board1.get_hashable_state()

    board2 = UTTTBoard()
    board2.make_move(0, 3, "X")
    board2.make_move(0, 4, "X")
    board2.make_move(0, 5, "X")  # Board 0 won by X
    hash2 = board2.get_hashable_state()

    assert (
        hash1 == hash2
    ), "Hash should be the same if winners are the same, regardless of cell values"
