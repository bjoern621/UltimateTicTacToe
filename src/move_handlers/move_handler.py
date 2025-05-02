from abc import ABC, abstractmethod
from typing import Tuple

from ttt_board import Player, CellIndex
from uttt_board import BoardIndex, UTTTBoard


class MoveHandler(ABC):
    """Abstract base class for move handlers."""

    def __init__(self, player: Player):
        self.player: Player = player

    @abstractmethod
    def get_move(
        self, board: UTTTBoard, forced_board: BoardIndex | None
    ) -> Tuple[BoardIndex, CellIndex]:
        """Calculates the next move."""
        pass
