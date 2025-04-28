from typing import List, Literal
from ttt_board import TTTBoard, Winner


type BoardIndex = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]


class UTTTBoard:
    def __init__(self):
        self.small_boards: List[TTTBoard] = [TTTBoard() for _ in range(9)]
        self.winner: Winner = None

    def get_small_board(self, index: int) -> TTTBoard:
        """Returns the Tic-Tac-Toe board at the given index (0-8)."""

        assert index >= 0, "Small board index must be non-negative."
        assert index < 9, "Small board index must be less than 9."

        return self.small_boards[index]

    def display_board(self):
        """Displays the Ultimate Tic-Tac-Toe board."""

        heavy_sep = "═══════╬═══════╬═══════"

        print()

        for big_row in range(3):
            for cell_row in range(5):
                small_board_rows: List[str] = []
                for big_col in range(3):
                    small_board_index = big_row * 3 + big_col
                    small_board = self.small_boards[small_board_index]

                    small_board_rows.append(small_board.get_row_string(cell_row))

                print(" " + " ║ ".join(small_board_rows) + " ")

            if big_row < 2:
                print(heavy_sep)

        print()

    # Add methods later for game logic:
    # def make_ultimate_move(self, board_index, cell_index, player): ...
    # def check_overall_win(self): ...
    # def determine_next_board(self, last_cell_index): ...
