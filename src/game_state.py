from move_handlers.move_handler import MoveHandler
from ttt_board import Player
from uttt_board import UTTTBoard


class GameState:
    """Class to manage the game state."""

    def __init__(self, board: UTTTBoard, playerX: MoveHandler, playerO: MoveHandler):
        self.board = board
        self.playerX = playerX
        self.playerO = playerO
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

            board_index, cell_index = (
                self.playerX.get_move(self.board)
                if self.current_player == "X"
                else self.playerO.get_move(self.board)
            )

            assert 0 <= board_index <= 8, "Board index must be between 0 and 8."
            assert 0 <= cell_index <= 8, "Cell index must be between 0 and 8."

            board = self.board.get_small_board(board_index)
            if board.winner is not None:
                print(f"Board {board_index} already won by {board.winner}! Try again.")
                continue

            if board.get_cell_value(cell_index) is not None:  # type: ignore
                print("Cell already occupied! Try again.")
                continue

            board.make_move(cell_index, self.current_player)  # type: ignore

            # TODO: Implement logic to check for overall win
            # TODO: Implement logic to determine next_board_index

            self.__switch_player()
