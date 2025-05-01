import math
import random
import time
from typing import Dict, List, Tuple, cast

from move_handlers.move_handler import MoveHandler

from ttt_board import Player, CellIndex, Winner
from uttt_board import BoardIndex, BoardStateHash, UTTTBoard


class MinimaxIterativeHandler(MoveHandler):
    """Minimax move handler for Ultimate Tic-Tac-Toe.\n
    It uses the Minimax algorithm with Alpha-Beta pruning to determine the best move.
    It is time-limited. It has a randomized exploration strategy. It uses memoization to speed up the search process. It stops searching when a winning/losing move is found.
    """

    # TODO: There is a bug searching for a move sometimes returns None for the move and a ?valid? score like 0, 10_000, 3.0, .... This is due to the memoization cache being used. See /bug.md.

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

    def __init__(self, player: Player, max_time: float = 1.0) -> None:
        super().__init__(player)
        self.__opponent: Player = "O" if player == "X" else "X"
        self.__max_time = max_time
        self.memo: Dict[
            Tuple[
                BoardStateHash, int, bool, BoardIndex | None
            ],  # [BoardStateHash, depth, is_maximizing_player, forced_board_index]
            Tuple[float, Tuple[BoardIndex, CellIndex] | None],  # [score, move]
        ] = {}

    def get_move(
        self, board: UTTTBoard, forced_board: BoardIndex | None
    ) -> tuple[BoardIndex, CellIndex]:
        """Calculates the best move using Minimax."""

        print(
            f"Iterative Minimax ({self.player}) thinking... Forced board: {forced_board}"
        )

        start_time = time.time()
        depth = 0
        best_move: Tuple[BoardIndex, CellIndex] | None = None
        best_score = -math.inf

        # Iterative deepening: increase depth until time limit is exceeded
        while True:
            depth += 1

            print(f"  Starting search at depth {depth}...")

            score, move, timed_out = self.__alphabeta_max(
                board, -math.inf, math.inf, depth, forced_board, start_time
            )

            if timed_out:
                print(f"  Time limit reached during depth {depth} search.")
                break

            print(f"  Finished! Found move: {move} with score: {score}.")

            if move is None:
                time.sleep(10)

            best_move = move
            best_score = score

            if abs(best_score) == self.__WINNING_SCORE:
                print(f"  Found guaranteed winning/losing move at depth {depth}.")
                break

        assert best_move is not None, "Minimax returned None for the move."

        print(
            f"Iterative Minimax ({self.player}) chose move: {best_move} with score: {best_score}. Took {time.time() - start_time:.2f} seconds. Cache size: {len(self.memo)}"
        )

        return best_move

    def __alphabeta_max(
        self,
        board: UTTTBoard,
        alpha: float,
        beta: float,
        depth: int,
        forced_board: BoardIndex | None,
        start_time: float,
    ) -> Tuple[float, Tuple[BoardIndex, CellIndex] | None, bool]:
        """Maximizing player for Alpha-Beta pruning."""

        # Time exceeded check
        if time.time() - start_time >= self.__max_time:
            return 0, None, True

        memo_key = (board.get_hashable_state(), depth, True, forced_board)
        if memo_key in self.memo:
            mem_score, mem_move = self.memo[memo_key]

            if mem_move is None:
                print(memo_key)
                print(self.memo[memo_key])

            return mem_score, mem_move, False

        if depth == 0:
            return self.__evaluate_board(board), None, False

        if board.winner is not None:
            return self.__evaluate_board(board), None, False

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

            score, _, timed_out = self.__alphabeta_min(
                board_copy, alpha, beta, depth - 1, next_forced_board, start_time
            )

            if timed_out:
                return 0, None, True

            if score > alpha:
                alpha = score
                best_move = (board_index, cell_index)

            if alpha >= beta:
                break  # Prune

        # best_move is None if all possible moves are explored and none are better than the current alpha value
        # if best_move is not None: # FIX: this fixed the bug where the move was None
        assert memo_key not in self.memo, "Memo key already exists."
        self.memo[memo_key] = (alpha, best_move)

        return alpha, best_move, False

    def __alphabeta_min(
        self,
        board: UTTTBoard,
        alpha: float,
        beta: float,
        depth: int,
        forced_board: BoardIndex | None,
        start_time: float,
    ) -> Tuple[float, Tuple[BoardIndex, CellIndex] | None, bool]:
        """Minimizing player for Alpha-Beta pruning."""

        # Time exceeded check
        if time.time() - start_time >= self.__max_time:
            return 0, None, True

        memo_key = (board.get_hashable_state(), depth, False, forced_board)
        if memo_key in self.memo:
            mem_score, mem_move = self.memo[memo_key]
            return mem_score, mem_move, False

        if depth == 0:
            return self.__evaluate_board(board), None, False

        if board.winner is not None:
            return self.__evaluate_board(board), None, False

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

            score, _, timed_out = self.__alphabeta_max(
                board_copy, alpha, beta, depth - 1, next_forced_board, start_time
            )

            if timed_out:
                return 0, None, True

            if score < beta:
                beta = score
                best_move = (board_index, cell_index)

            if alpha >= beta:
                break  # Prune

        # best_move is None if all possible moves are explored and none are better than the current alpha value
        # if best_move is not None:
        assert memo_key not in self.memo, "Memo key already exists."
        self.memo[memo_key] = (beta, best_move)

        return beta, best_move, False

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

        # 1. Check for terminal states (win, loss, draw)
        if board.winner == self.player:
            return self.__WINNING_SCORE
        if board.winner == self.__opponent:
            return -self.__WINNING_SCORE
        if board.winner == "Draw":
            return 0

        score = 0.0

        # 2. Evaluate the UTTT board based on small board winners
        score += self.__evaluate_uttt_board(board)

        # 3. Add bonus for winning individual small boards (only really useful for sudden death when the game is over and no player has three in a row)
        score += self.__evaluate_individual_boards(board)

        # 4. Add heuristic based on lines within (playable) small boards
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
        """Evaluates the UTTT board based on the winners of the small boards. +/- 150 for 2 small boards with potential for completion, +/- 50 for 1 small board with potential for completion."""

        score = 0.0

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
