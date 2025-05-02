import math
import random
import time
from typing import List, Tuple, cast

from move_handlers.move_handler import MoveHandler

from ttt_board import Player, CellIndex, Winner
from uttt_board import BoardIndex, UTTTBoard


class MinimaxHandler(MoveHandler):
    """Minimax move handler for Ultimate Tic-Tac-Toe.\n
    It uses the Minimax algorithm with Alpha-Beta pruning to determine the best move.
    It is depth-limited. It has a randomized exploration strategy."""

    __WINNING_LINES = [
        (0, 1, 2),  # Rows
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),  # Columns
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),  # Diagonals
        (2, 4, 6),
    ]

    def __init__(
        self,
        player: Player,
        max_depth: int = 5,
    ) -> None:
        super().__init__(player)
        self.__opponent: Player = "O" if player == "X" else "X"
        self.__max_depth = max_depth
        self.runtime = 0.0

    def get_move(
        self, board: UTTTBoard, forced_board: BoardIndex | None
    ) -> tuple[BoardIndex, CellIndex]:
        """Calculates the best move using Minimax."""

        print(f"Minimax ({self.player}) thinking... Forced board: {forced_board}")

        start_time = time.time()

        score, move = self.__alphabeta_max(
            board, -math.inf, math.inf, self.__max_depth, forced_board
        )

        assert move is not None, "Minimax returned None for the move."

        print(
            f"Minimax ({self.player}) chose move: {move} with score: {score}. Took {time.time() - start_time:.2f} seconds."
        )

        self.runtime += time.time() - start_time

        return move

    def __alphabeta_max(
        self,
        board: UTTTBoard,
        alpha: float,
        beta: float,
        depth: int,
        forced_board: BoardIndex | None,
    ) -> Tuple[float, Tuple[BoardIndex, CellIndex] | None]:
        """Maximizing player for Alpha-Beta pruning."""

        if depth == 0:
            return self.__evaluate_board(board), None

        if board.winner is not None:
            return self.__evaluate_board(board), None

        possible_moves = self.__get_valid_moves(board, forced_board)

        assert possible_moves, "No possible moves available."

        random.shuffle(possible_moves)

        # Initialize best_move to None; it will be updated when valid moves are evaluated.
        best_move: Tuple[BoardIndex, CellIndex] | None = None

        for board_index, cell_index in possible_moves:
            board_copy = board.copy()
            board_copy.make_move(board_index, cell_index, self.player)
            next_forced_board = self.__determine_next_forced_board(
                board_copy, cell_index
            )

            score, _ = self.__alphabeta_min(
                board_copy, alpha, beta, depth - 1, next_forced_board
            )

            if score > alpha:
                alpha = score
                best_move = (board_index, cell_index)

            if alpha >= beta:
                break  # Prune

        return alpha, best_move

    def __alphabeta_min(
        self,
        board: UTTTBoard,
        alpha: float,
        beta: float,
        depth: int,
        forced_board: BoardIndex | None,
    ) -> Tuple[float, Tuple[BoardIndex, CellIndex] | None]:
        """Minimizing player for Alpha-Beta pruning."""

        if depth == 0:
            return self.__evaluate_board(board), None

        if board.winner is not None:
            return self.__evaluate_board(board), None

        possible_moves = self.__get_valid_moves(board, forced_board)

        assert possible_moves, "No possible moves available."

        random.shuffle(possible_moves)

        # Initialize best_move to None; it will be updated when valid moves are evaluated.
        best_move: Tuple[BoardIndex, CellIndex] | None = None

        for board_index, cell_index in possible_moves:
            board_copy = board.copy()
            board_copy.make_move(board_index, cell_index, self.__opponent)
            next_forced_board = self.__determine_next_forced_board(
                board_copy, cell_index
            )

            score, _ = self.__alphabeta_max(
                board_copy, alpha, beta, depth - 1, next_forced_board
            )

            if score < beta:
                beta = score
                best_move = (board_index, cell_index)

            if alpha >= beta:
                break  # Prune

        return beta, best_move

    def __get_valid_moves(
        self, board: UTTTBoard, forced_board: BoardIndex | None
    ) -> List[Tuple[BoardIndex, CellIndex]]:
        """Gets all valid moves for the current state."""

        moves: List[Tuple[BoardIndex, CellIndex]] = []
        possible_board_indices: List[BoardIndex] = []

        if forced_board is None:
            # Determine all playable boards if not forced to a specific one
            possible_board_indices = [
                cast(BoardIndex, board_index)
                for board_index in range(9)
                if board.get_small_board(cast(BoardIndex, board_index)).winner is None
            ]
        else:
            # Only check the forced board
            possible_board_indices = [forced_board]

        for board_index in possible_board_indices:
            small_board = board.get_small_board(board_index)

            assert small_board.winner is None, "Small board should be playable."

            for cell_index_int in range(9):
                cell_index = cast(CellIndex, cell_index_int)

                if small_board.get_cell_value(cell_index) is None:
                    moves.append((board_index, cell_index))

        return moves

    def __determine_next_forced_board(
        self, board: UTTTBoard, cell_index: CellIndex
    ) -> BoardIndex | None:
        """Determines the index of the small board the next player is forced to play in, based on the cell_index of the current move."""

        next_board_index: BoardIndex = cell_index
        next_small_board = board.get_small_board(next_board_index)

        if next_small_board.winner is not None:
            return None
        else:
            return next_board_index

    def __evaluate_board(self, board: UTTTBoard) -> float:
        """Heuristic evaluation function for the UTTT board state."""

        score = 0.0

        # 1. Evaluate the UTTT board based on small board winners
        score += self.__evaluate_uttt_board(board)

        # 2. Add bonus for winning individual small boards (only really useful for sudden death when the game is over and no player has three in a row)
        score += self.__evaluate_individual_boards(board)

        # 3. Add heuristic based on lines within (playable) small boards
        score += self.__evaluate_small_boards(board)

        return score

    def __evaluate_individual_boards(self, board: UTTTBoard) -> float:
        """Evaluates the individual small boards based on their winners. +/- 10 for each small board won."""

        score = 0.0

        for i in range(9):
            small_winner = board.get_small_board(cast(BoardIndex, i)).winner
            if small_winner == self.player:
                score += 10
            elif small_winner == self.__opponent:
                score -= 10

        return score

    def __evaluate_uttt_board(self, board: UTTTBoard) -> float:
        """Evaluates the UTTT board based on the winners of the small boards. +/- 1_000 for an overall win/loss, +/- 150 for 2 small boards with potential for completion, +/- 50 for 1 small board with potential for completion."""

        score = 0.0

        if board.winner == self.player:
            score += 1000
        if board.winner == self.__opponent:
            score -= 1000
        if board.winner == "Draw":
            score += 0

        small_board_winners: List[Winner] = [
            board.get_small_board(cast(BoardIndex, i)).winner for i in range(9)
        ]

        for combo in self.__WINNING_LINES:
            line = (
                small_board_winners[combo[0]],
                small_board_winners[combo[1]],
                small_board_winners[combo[2]],
            )
            score += self.__score_line(line, 150, 50)

        return score

    def __score_line(
        self,
        line: Tuple[Winner, Winner, Winner],
        two_in_a_row: float,
        one_in_a_row: float,
    ) -> float:
        """Assigns a score to a single line (row, col, or diag) of winners. +/- two_in_a_row for 2 cells with potential for completion, +/- one_in_a_row for 1 cell with potential for completion."""

        my_cells = line.count(self.player)
        op_cells = line.count(self.__opponent)
        empty_cells = line.count(None)

        score = 0.0

        if my_cells == 2 and empty_cells == 1:
            score = two_in_a_row
        elif op_cells == 2 and empty_cells == 1:
            score = -two_in_a_row
        elif my_cells == 1 and empty_cells == 2:
            score = one_in_a_row
        elif op_cells == 1 and empty_cells == 2:
            score = -one_in_a_row

        return score

    def __evaluate_small_boards(self, board: UTTTBoard) -> float:
        """Evaluates the playable small boards based on their lines. +/- 1.5 for 2 cells with potential for completion, +/- 0.5 for 1 cell with potential for completion."""

        score = 0.0

        for i in range(9):
            board_index = cast(BoardIndex, i)
            small_board = board.get_small_board(board_index)
            if small_board.winner is None:
                for combo in self.__WINNING_LINES:
                    line = (
                        small_board.get_cell_value(cast(CellIndex, combo[0])),
                        small_board.get_cell_value(cast(CellIndex, combo[1])),
                        small_board.get_cell_value(cast(CellIndex, combo[2])),
                    )
                    score += self.__score_line(line, 1.5, 0.5)

        return score
