from ttt_board import Player
from uttt_board import UTTTBoard


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

                if not (0 <= board_index <= 8):
                    print("Board index must be between 0 and 8.")
                    continue

                if not (0 <= cell_index <= 8):
                    print("Cell index must be between 0 and 8.")
                    continue

                board = self.board.get_small_board(board_index)
                if board.winner is not None:
                    print(
                        f"Board {board_index} already won by {board.winner}! Try again."
                    )
                    continue

                if board.get_cell_value(cell_index) is not None:  # type: ignore
                    print("Cell already occupied! Try again.")
                    continue

                board.make_move(cell_index, self.current_player)  # type: ignore

                # TODO: Implement logic to check for overall win
                # TODO: Implement logic to determine next_board_index

                self.__switch_player()

            except ValueError:
                print(
                    "Invalid input! Please enter two numbers separated by space (e.g., '0 4')."
                )
                continue
            except Exception as e:
                print(
                    f"An unexpected error occurred: {e}"
                )  # Catch other potential errors
                continue
