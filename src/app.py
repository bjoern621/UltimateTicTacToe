from typing import List, Literal


type Player = Literal["X", "O"]
type Winner = Player | Literal["Draw"] | None
type Square = Player | None

# Define the 5x5 ASCII art patterns
X_ART = [
    "╲   ╱",
    " ╲ ╱ ",
    "  ╳  ",
    " ╱ ╲ ",
    "╱   ╲",
]

O_ART = [
    "╭───╮",
    "│   │",
    "│   │",
    "│   │",
    "╰───╯",
]

# DRAW_ART = [
#     "░░░░░",
#     "░▓▓▓░",
#     "░▓░▓░",
#     "░▓▓▓░",
#     "░░░░░",
# ]

DRAW_ART = [
    "# # #",
    " # # ",
    "# # #",
    " # # ",
    "# # #",
]

LIGHT_SEP = "─┼─┼─"


class TTTBoard:
    def __init__(self):
        self.board: List[Square] = [None] * 9
        self.winner: Winner = None

    def get_row_string(self, row_index: int) -> str:
        """Returns a string representation of a specific row (0-4) for the small board."""

        assert 0 <= row_index <= 4, "Row index must be between 0 and 4."

        if self.winner == "X":
            return X_ART[row_index]
        if self.winner == "O":
            return O_ART[row_index]
        if self.winner == "Draw":
            return DRAW_ART[row_index]

        # If no winner, render the board state and separators
        if row_index == 1 or row_index == 3:
            return LIGHT_SEP  # Return separator for rows 1 and 3

        # Map 0, 2, 4 to board rows 0, 1, 2
        board_row_index = row_index // 2

        def format_cell(cell_value: Square) -> str:
            return cell_value if cell_value is not None else " "

        start_cell_index = board_row_index * 3
        c1 = format_cell(self.board[start_cell_index])
        c2 = format_cell(self.board[start_cell_index + 1])
        c3 = format_cell(self.board[start_cell_index + 2])
        return f"{c1}│{c2}│{c3}"

    # Add methods later to check win/draw for this small board
    # def check_win(self): ...
    # def check_draw(self): ...
    # def make_move(self, position, player): ...


class UTTTBoard:
    def __init__(self):
        self.small_boards: List[TTTBoard] = [TTTBoard() for _ in range(9)]
        self.winner: Winner = None
        # Next board to play in, None is any board
        self.next_board_index: int | None = None

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

                    # Pad the string to ensure consistent width (5 chars)
                    # small_board_rows.append(
                    #     f"{board.get_row_string(cell_row):^5}"
                    # )  # Center align and pad

                    small_board_rows.append(small_board.get_row_string(cell_row))

                print(" " + " ║ ".join(small_board_rows) + " ")

            if big_row < 2:
                print(heavy_sep)

        print()

    # Add methods later for game logic:
    # def make_ultimate_move(self, board_index, cell_index, player): ...
    # def check_overall_win(self): ...
    # def determine_next_board(self, last_cell_index): ...


def main():
    print("Running...")

    board = UTTTBoard()

    # Example: Make a move in the top-left small board (index 0), at its center cell (index 4)
    # ultimate_game.get_small_board(0).board[4] = 'X'
    # ultimate_game.next_board_index = 4 # Next move must be in board 4

    print("\nInitial empty ultimate board:")
    board.display_board()

    # You would add game loop logic here


if __name__ == "__main__":
    main()
