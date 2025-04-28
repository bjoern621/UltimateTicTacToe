from move_handlers.move_handler import MoveHandler
from ttt_board import CellIndex
from uttt_board import BoardIndex, UTTTBoard
import random


class RandomHandler(MoveHandler):
    def get_move(self, board: UTTTBoard) -> tuple[BoardIndex, CellIndex]:
        board_index: BoardIndex = random.randint(0, 8)  # type: ignore
        cell_index: CellIndex = random.randint(0, 8)  # type: ignore

        return (board_index, cell_index)
