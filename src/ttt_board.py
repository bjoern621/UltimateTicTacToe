from typing import List, Literal


type Player = Literal["X", "O"]
type Winner = Player | Literal["Draw"] | None
type CellValue = Player | None
type CellIndex = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]


class TTTBoard:
    """
    Represents a single 3x3 Tic-Tac-Toe board within a larger Ultimate Tic-Tac-Toe game.
    This class manages the state of a small 3x3 board, including the marks ('X' or 'O')
    in each cell and whether the board has been won or resulted in a draw. It also
    provides functionality to render a string representation of the board, including
    ASCII art for displaying the final state (X win, O win, or Draw).
    """

    __X_ART = [
        "╲   ╱",
        " ╲ ╱ ",
        "  ╳  ",
        " ╱ ╲ ",
        "╱   ╲",
    ]

    __O_ART = [
        "╭───╮",
        "│   │",
        "│   │",
        "│   │",
        "╰───╯",
    ]

    __DRAW_ART = [
        "# # #",
        " # # ",
        "# # #",
        " # # ",
        "# # #",
    ]

    __LIGHT_SEP = "─┼─┼─"

    __YELLOW = "\033[33m"

    __HIGHLIGHT_COLOR = __YELLOW
    __RESET = "\033[0m"

    def __init__(self):
        self.__board: List[CellValue] = [None] * 9
        self.winner: Winner = None

    def get_row_string(self, row_index: int, is_forced_board: bool) -> str:
        """Returns a string representation of a specific row (0-4) for the small board. The TTT board fits in a 5x5 grid."""

        assert 0 <= row_index <= 4, "Row index must be between 0 and 4."

        if self.winner == "X":
            return self.__X_ART[row_index]
        if self.winner == "O":
            return self.__O_ART[row_index]
        if self.winner == "Draw":
            return self.__DRAW_ART[row_index]

        # If no winner, render the board state and separators

        if row_index == 1 or row_index == 3:
            if is_forced_board:
                return f"{self.__HIGHLIGHT_COLOR}{self.__LIGHT_SEP}{self.__RESET}"
            else:
                return self.__LIGHT_SEP

        # Map 0, 2, 4 to board rows 0, 1, 2
        board_row_index = row_index // 2

        def format_cell(cell_value: CellValue) -> str:
            return cell_value if cell_value is not None else " "

        start_cell_index = board_row_index * 3
        c1 = format_cell(self.__board[start_cell_index])
        c2 = format_cell(self.__board[start_cell_index + 1])
        c3 = format_cell(self.__board[start_cell_index + 2])

        if is_forced_board:
            colored_seperator = f"{self.__HIGHLIGHT_COLOR}│{self.__RESET}"
            return f"{c1}{colored_seperator}{c2}{colored_seperator}{c3}"
        else:
            return f"{c1}│{c2}│{c3}"

    def make_move(self, index: CellIndex, value: Player) -> None:
        """Sets the value of a cell in the small board and checks for a winner."""

        assert self.__board[index] is None, "Cell already occupied."
        assert self.winner is None, "Game already won."

        self.__board[index] = value

        winner = self.__check_winner()
        if winner != None:
            self.winner = winner

    def __check_winner(self) -> Winner:
        """Checks for a winner in this board."""

        lines = [
            [self.__board[i] for i in (0, 1, 2)],  # Top Row
            [self.__board[i] for i in (3, 4, 5)],  # Middle Row
            [self.__board[i] for i in (6, 7, 8)],  # Bottom Row
            [self.__board[i] for i in (0, 3, 6)],  # Left Column
            [self.__board[i] for i in (1, 4, 7)],  # Middle Column
            [self.__board[i] for i in (2, 5, 8)],  # Right Column
            [self.__board[i] for i in (0, 4, 8)],  # Diagonal \
            [self.__board[i] for i in (2, 4, 6)],  # Diagonal /
        ]

        for line in lines:
            if all(cell == "X" for cell in line):
                return "X"
            if all(cell == "O" for cell in line):
                return "O"

        if all(cell is not None for cell in self.__board):
            return "Draw"

        return None

    def get_cell_value(self, index: CellIndex) -> CellValue:
        """Returns the value of a cell in the small board."""

        return self.__board[index]

    def copy(self) -> "TTTBoard":
        """Creates a deep copy of the TTTBoard."""

        new_board = TTTBoard()
        new_board.__board = self.__board[:]
        new_board.winner = self.winner
        return new_board

    def get_cells(self) -> tuple[CellValue, ...]:
        """Returns the current state of the board cells as a tuple."""
        return tuple(self.__board)
