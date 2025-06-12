import math
import random
import time
from typing import List, Tuple, cast, Dict
import itertools

from move_handlers.move_handler import MoveHandler

from ttt_board import CellValue, Player, CellIndex, Winner
from uttt_board import BoardIndex, UTTTBoard


class MinimaxHandler(MoveHandler):
    """Minimax move handler for Ultimate Tic-Tac-Toe.\n
    It uses the Minimax algorithm with Alpha-Beta pruning to determine the best move.
    It is depth-limited. It has a randomized exploration strategy."""

    __name__ = "Minimax"

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

    __WINNING_SCORE = 10_000

    def __init__(
        self,
        player: Player,
        max_depth: int = 5,
        log: bool = True
    ) -> None:
        super().__init__(player, log)
        self.__opponent: Player = "O" if player == "X" else "X"
        self.__max_depth = max_depth
        self.evaluation_time = 0.0
        self.uttt_board_eval_time = 0.0
        self.individual_boards_eval_time = 0.0
        self.small_boards_eval_time = 0.0

        self.__precomputed_line_counts: Dict[
            Tuple[Winner, ...], Tuple[int, int, int, int]
        ] = self.__precompute_line_counts()

        self.__name__  = f"Minimax (Max_Depth: {max_depth})"

    def get_move(
        self, board: UTTTBoard, forced_board: BoardIndex | None
    ) -> tuple[BoardIndex, CellIndex]:
        return self.get_move_confidence(board, forced_board)[0]
    
    def get_move_confidence(
            self, board: UTTTBoard, forced_board: BoardIndex | None
    ) -> Tuple[Tuple[BoardIndex, CellIndex], float]:
        """Calculates the best move using Minimax."""

        if self.log: print(f"Minimax ({self.player}) thinking... Forced board: {forced_board}")

        start_time = time.time()
        self.evaluation_time = 0.0
        self.uttt_board_eval_time = 0.0
        self.individual_boards_eval_time = 0.0
        self.small_boards_eval_time = 0.0

        score, move = self.__alphabeta_max(
            board, -math.inf, math.inf, self.__max_depth, forced_board
        )

        assert move is not None, "Minimax returned None for the move."

        if self.log: print(
            f"Minimax ({self.player}) chose move: {move} with score: {score}. Took {time.time() - start_time:.2f} seconds "
            f"(Eval time: {self.evaluation_time:.2f}s - UTTT: {self.uttt_board_eval_time:.2f}s, "
            f"Indiv: {self.individual_boards_eval_time:.2f}s, Small: {self.small_boards_eval_time:.2f}s)."
        )

        return (move, score)
        

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
        eval_start_time = time.time()

        if board.winner == self.player:
            return self.__WINNING_SCORE
        if board.winner == self.__opponent:
            return -self.__WINNING_SCORE
        if board.winner == "Draw":
            return 0

        score = 0.0

        # 1. Evaluate the UTTT board based on small board winners
        uttt_start_time = time.time()
        score += self.__evaluate_uttt_board_lines(board)
        self.uttt_board_eval_time += time.time() - uttt_start_time

        # 2. Add bonus for winning individual small boards
        indiv_start_time = time.time()
        score += self.__evaluate_individual_boards(board)
        self.individual_boards_eval_time += time.time() - indiv_start_time

        # 3. Add heuristic based on lines within (playable) small boards
        small_start_time = time.time()
        score += self.__evaluate_small_boards(board)
        self.small_boards_eval_time += time.time() - small_start_time

        self.evaluation_time += time.time() - eval_start_time
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

    def __evaluate_uttt_board_lines(self, board: UTTTBoard) -> float:
        """Evaluates the UTTT board based on the winners of the small boards. +/- 150 for 2 small boards with potential for completion, +/- 50 for 1 small board with potential for completion."""

        winners_tuple: Tuple[Winner, ...] = tuple(
            board.get_small_board(cast(BoardIndex, i)).winner for i in range(9)
        )

        my_two, op_two, my_one, op_one = self.__precomputed_line_counts.get(
            winners_tuple, (-1, -1, -1, -1)
        )

        if my_two == -1:
            # This means the board state is not precomputed, which is the case if there is at lest one Draw in the tuple.
            # In this case we manually compute the score.
            # A drawn small board is surprisingly rare in the game, so this is not a performance issue.

            my_two = 0
            op_two = 0
            my_one = 0
            op_one = 0

            for combo in self.__WINNING_LINES:
                line: Tuple[Winner, Winner, Winner] = (
                    winners_tuple[combo[0]],
                    winners_tuple[combo[1]],
                    winners_tuple[combo[2]],
                )

                my_cells = line.count(self.player)
                op_cells = line.count(self.__opponent)
                empty_cells = line.count(None)  # Ignore Draws

                if my_cells == 2 and empty_cells == 1:
                    my_two += 1
                elif op_cells == 2 and empty_cells == 1:
                    op_two += 1
                elif my_cells == 1 and empty_cells == 2:
                    my_one += 1
                elif op_cells == 1 and empty_cells == 2:
                    op_one += 1

        assert my_two != -1, "my_two should not be -1 after evaluation."

        two_in_a_row = 150
        one_in_a_row = 50

        score = (
            (my_two * two_in_a_row)
            + (op_two * -two_in_a_row)
            + (my_one * one_in_a_row)
            + (op_one * -one_in_a_row)
        )

        return score

    def __precompute_line_counts(
        self,
    ) -> Dict[Tuple[Winner, ...], Tuple[int, int, int, int]]:
        """Precomputes counts of potential winning lines for all possible 9 cell states. There are 3^9 = 19683 possible combinations.
        Returns a dictionary mapping board state tuples to a tuple:
        (my_two, op_two, my_one, op_one)
        where:
        - my_two: lines with 2 of player's marks and 1 empty
        - op_two: lines with 2 of opponent's marks and 1 empty
        - my_one: lines with 1 of player's mark and 2 empty
        - op_one: lines with 1 of opponent's mark and 2 empty
        """

        line_counts: Dict[Tuple[Winner, ...], Tuple[int, int, int, int]] = {}
        possible_values: List[CellValue] = [self.player, self.__opponent, None]

        for board_tuple in itertools.product(possible_values, repeat=9):
            my_two_count = 0
            op_two_count = 0
            my_one_count = 0
            op_one_count = 0

            for combo in self.__WINNING_LINES:
                line: Tuple[CellValue, CellValue, CellValue] = (
                    board_tuple[combo[0]],
                    board_tuple[combo[1]],
                    board_tuple[combo[2]],
                )

                my_cells = line.count(self.player)
                op_cells = line.count(self.__opponent)
                empty_cells = line.count(None)

                if my_cells == 2 and empty_cells == 1:
                    my_two_count += 1
                elif op_cells == 2 and empty_cells == 1:
                    op_two_count += 1
                elif my_cells == 1 and empty_cells == 2:
                    my_one_count += 1
                elif op_cells == 1 and empty_cells == 2:
                    op_one_count += 1

            line_counts[board_tuple] = (
                my_two_count,
                op_two_count,
                my_one_count,
                op_one_count,
            )

        return line_counts

    def __evaluate_small_boards(self, board: UTTTBoard) -> float:
        """Evaluates the playable small boards based on their lines. +/- 1.5 for 2 cells with potential for completion, +/- 0.5 for 1 cell with potential for completion."""

        score = 0.0
        two_in_a_row = 1.5
        one_in_a_row = 0.5

        for i in range(9):
            board_index = cast(BoardIndex, i)
            small_board = board.get_small_board(board_index)
            if small_board.winner is None:
                board_tuple: Tuple[CellValue, ...] = small_board.get_cells()
                my_two, op_two, my_one, op_one = self.__precomputed_line_counts[
                    board_tuple
                ]

                score += (
                    (my_two * two_in_a_row)
                    + (op_two * -two_in_a_row)
                    + (my_one * one_in_a_row)
                    + (op_one * -one_in_a_row)
                )
        return score
