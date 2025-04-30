from move_handlers.move_handler import MoveHandler
from ttt_board import CellIndex
from uttt_board import BoardIndex, UTTTBoard
import random


class RandomHandler(MoveHandler):
    def get_move(
        self, board: UTTTBoard, forced_board: BoardIndex | None
    ) -> tuple[BoardIndex, CellIndex]:
        if forced_board is not None:
            board_index = forced_board
        else:
            board_index: BoardIndex = random.randint(0, 8)  # type: ignore

            while board.get_small_board(board_index).winner is not None:
                board_index = random.randint(0, 8)  # type: ignore

        cell_index: CellIndex = random.randint(0, 8)  # type: ignore

        while board.get_small_board(board_index).get_cell_value(cell_index) is not None:  # type: ignore
            cell_index = random.randint(0, 8)  # type: ignore

        return (board_index, cell_index)
