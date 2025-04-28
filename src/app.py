from typing import List, Literal


type Player = Literal["X", "O"]
type Winner = Player | Literal["Draw"] | None
type Square = Player | None


class TTTBoard:
    def __init__(self):
        self.board: List[Square] = [None] * 9
        self.winner: Winner = None

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

        def format_cell(cell_value: Square) -> str:
            return cell_value if cell_value is not None else " "

        light_sep = "─┼─┼─"
        heavy_sep = "═══════╬═══════╬═══════"

        print()

        for big_row in range(3):
            for cell_row in range(3):
                small_board_rows: List[str] = []
                for big_col in range(3):
                    small_board_index = big_row * 3 + big_col
                    board = self.small_boards[small_board_index].board
                    start_cell_index = cell_row * 3
                    c1 = format_cell(board[start_cell_index])
                    c2 = format_cell(board[start_cell_index + 1])
                    c3 = format_cell(board[start_cell_index + 2])
                    small_board_rows.append(f"{c1}│{c2}│{c3}")

                print(" " + " ║ ".join(small_board_rows) + " ")

                # Print light horizontal separator within a big row
                if cell_row < 2:
                    print(f" {light_sep} ║ {light_sep} ║ {light_sep} ")

            # Print heavy horizontal separator between big rows
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
