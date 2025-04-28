from move_handlers.move_handler import MoveHandler
from ttt_board import CellIndex, Player
from uttt_board import BoardIndex, UTTTBoard


class ConsoleHandler(MoveHandler):
    def __init__(self, player: Player):
        self.player = player

    def get_move(
        self, board: UTTTBoard, forced_board: BoardIndex | None
    ) -> tuple[BoardIndex, CellIndex]:
        if forced_board is not None:
            print(f"Player {self.player}, you must play in board {forced_board}.")
            board_index = forced_board

            try:
                player_input = input(
                    f"Player {self.player}, enter your move (cell index): "
                )
            except KeyboardInterrupt:
                exit(0)

            try:
                cell_index = int(player_input)
            except ValueError:
                print("Invalid input. Please enter one integer (e.g., '4').")
                return self.get_move(board, forced_board)
        else:
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
                return self.get_move(board, forced_board)

        if not (0 <= board_index <= 8):
            print("Board index must be between 0 and 8.")
            return self.get_move(board, forced_board)

        if not (0 <= cell_index <= 8):
            print("Cell index must be between 0 and 8.")
            return self.get_move(board, forced_board)

        return (board_index, cell_index)  # type: ignore
