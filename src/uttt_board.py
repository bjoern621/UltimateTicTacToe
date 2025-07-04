from typing import List, Literal, Tuple, Union
from ttt_board import CellIndex, CellValue, Player, TTTBoard, Winner


type BoardIndex = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]
type BoardStateHash = Tuple[Union[Winner, Tuple[CellValue, ...]], ...]


class UTTTBoard:
    def __init__(self):
        self.__small_boards: List[TTTBoard] = [TTTBoard() for _ in range(9)]
        self.winner: Winner = None

    def get_small_board(self, index: BoardIndex) -> TTTBoard:
        """Returns the Tic-Tac-Toe board at the given index (0-8)."""

        return self.__small_boards[index]

    def display_board(self, current_forced_board_index: BoardIndex | None) -> None:
        """Displays the Ultimate Tic-Tac-Toe board."""

        heavy_sep = "═══════╬═══════╬═══════"

        print()

        for big_row in range(3):
            for cell_row in range(5):
                small_board_rows: List[str] = []
                for big_col in range(3):
                    small_board_index = big_row * 3 + big_col
                    small_board = self.__small_boards[small_board_index]

                    small_board_rows.append(
                        small_board.get_row_string(
                            cell_row, small_board_index == current_forced_board_index
                        )
                    )

                print(" " + " ║ ".join(small_board_rows) + " ")

            if big_row < 2:
                print(heavy_sep)

        print()

    def make_move(
        self, board_index: BoardIndex, cell_index: CellIndex, player: Player
    ) -> None:
        """Makes a move on the specified small board and checks for a winner."""

        small_board = self.__small_boards[board_index]

        small_board.make_move(cell_index, player)

        if small_board.winner is not None:
            self.winner = self.__check_winner()

    def __check_winner(self) -> Winner:
        """Checks if the entire UTTT board has a winner."""

        lines = [
            (0, 1, 2),  # Top Row
            (3, 4, 5),  # Middle Row
            (6, 7, 8),  # Bottom Row
            (0, 3, 6),  # Left Column
            (1, 4, 7),  # Middle Column
            (2, 5, 8),  # Right Column
            (0, 4, 8),  # Diagonal \
            (2, 4, 6),  # Diagonal /
        ]

        for line in lines:
            if all(
                self.__small_boards[cell_index].winner == "X" for cell_index in line
            ):
                return "X"
            if all(
                self.__small_boards[cell_index].winner == "O" for cell_index in line
            ):
                return "O"

        if all(board.winner is not None for board in self.__small_boards):
            return "Draw"

        return None

    def copy(self) -> "UTTTBoard":
        """Creates a deep copy of the UTTTBoard."""

        new_board = UTTTBoard()
        new_board.__small_boards = [
            small_board.copy() for small_board in self.__small_boards
        ]
        new_board.winner = self.winner
        return new_board

    def get_hashable_state(
        self,
    ) -> BoardStateHash:
        """
        Returns a hashable representation of the current board state.
        The hash is a tuple of length 9. Each element corresponds to a small board.
        If the small board has a winner, the element is the Winner ('X', 'O', 'Draw').
        If the small board has no winner, the element is a tuple of the 9 CellValues.
        Can be used for memoization.
        """

        state_list: List[Union[Winner, Tuple[CellValue, ...]]] = []
        for board in self.__small_boards:
            if board.winner is not None:
                state_list.append(board.winner)
            else:
                state_list.append(tuple(board._TTTBoard__board))  # type: ignore

        return tuple(state_list)
