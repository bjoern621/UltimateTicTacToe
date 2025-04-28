from move_handlers.move_handler import MoveHandler
from ttt_board import CellIndex, Player
from uttt_board import BoardIndex, UTTTBoard


class ConsoleHandler(MoveHandler):
    def __init__(self, player: Player):
        self.player = player

    def get_move(self, board: UTTTBoard) -> tuple[BoardIndex, CellIndex]:
        try:
            player_input = input(
                f"Player {self.player}, enter your move (board index and cell index): "
            )
        except KeyboardInterrupt:
            exit(0)

        try:
            board_index, cell_index = map(int, player_input.split())
        except ValueError:
            print(
                "Invalid input. Please enter two integers separated by a space (e.g., '0 4')."
            )
            return self.get_move(board)

        if not (0 <= board_index <= 8):
            print("Board index must be between 0 and 8.")
            return self.get_move(board)

        if not (0 <= cell_index <= 8):
            print("Cell index must be between 0 and 8.")
            return self.get_move(board)

        return (board_index, cell_index)  # type: ignore
