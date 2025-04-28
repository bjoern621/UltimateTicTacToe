from abc import ABC, abstractmethod

from ttt_board import CellIndex
from uttt_board import BoardIndex, UTTTBoard


class MoveHandler(ABC):
    @abstractmethod
    def get_move(self, board: UTTTBoard) -> tuple[BoardIndex, CellIndex]:
        pass
