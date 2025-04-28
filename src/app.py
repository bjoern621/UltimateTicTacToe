from typing import List, Literal


type Player = Literal["X", "O"]
type Winner = Player | Literal["Draw"] | None
type Square = Player | None
type CellIndex = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]
type BoardIndex = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]


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

    def __init__(self):
        self.__board: List[Square] = [None] * 9
        self.winner: Winner = None

    def get_row_string(self, row_index: int) -> str:
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
            return self.__LIGHT_SEP

        # Map 0, 2, 4 to board rows 0, 1, 2
        board_row_index = row_index // 2

        def format_cell(cell_value: Square) -> str:
            return cell_value if cell_value is not None else " "

        start_cell_index = board_row_index * 3
        c1 = format_cell(self.__board[start_cell_index])
        c2 = format_cell(self.__board[start_cell_index + 1])
        c3 = format_cell(self.__board[start_cell_index + 2])
        return f"{c1}│{c2}│{c3}"

    def make_move(self, index: CellIndex, value: Player) -> None:
        """Sets the value of a cell in the small board."""

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

    def get_cell_value(self, index: CellIndex) -> Square:
        """Returns the value of a cell in the small board."""

        assert 0 <= index < 9, "Cell index must be between 0 and 8."

        return self.__board[index]


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


class GameState:
    """Class to manage the game state."""

    def __init__(self, board: UTTTBoard):
        self.board = board
        self.current_player: Player = "X"
        self.game_over: bool = False

        # Next board to play in, None is any board
        self.next_board_index: int | None = None
        self.next_player: Player = "X"

    def __switch_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"

    def run_game_loop(self):
        """Main game loop."""

        while not self.game_over:
            self.board.display_board()

            # Get the current player's move
            move = input(
                f"Player {self.current_player}, enter your move (board index and cell index): "
            )
            try:
                board_index, cell_index = map(int, move.split())

                board = self.board.get_small_board(board_index)
                if board.winner is not None:
                    print(f"Board {board_index} already won by {board.winner}!")
                    continue

                if cell_index < 0 or cell_index > 8:
                    print("Cell index must be between 0 and 8.")
                    continue

                if board.get_cell_value(cell_index) is not None:  # type: ignore
                    print("Cell already occupied! Try again.")
                    continue

                self.board.get_small_board(board_index).make_move(
                    cell_index, self.current_player  # type: ignore
                )

                self.__switch_player()

            except (ValueError, IndexError):
                print("Invalid input! Please enter valid indices.")
                continue


def main():
    print("Running...")

    board = UTTTBoard()

    # Example: Make a move in the top-left small board (index 0), at its center cell (index 4)
    # ultimate_game.get_small_board(0).board[4] = 'X'
    # ultimate_game.next_board_index = 4 # Next move must be in board 4

    print("\nInitial empty ultimate board:")
    board.display_board()

    state = GameState(board)

    state.run_game_loop()

    # You would add game loop logic here


if __name__ == "__main__":
    main()
