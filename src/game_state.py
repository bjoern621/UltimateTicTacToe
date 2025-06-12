from typing import List, Tuple
from move_handlers.minimax_handler import MinimaxHandler
from move_handlers.random_handler import RandomHandler
from ttt_board import Player
from uttt_board import BoardIndex, UTTTBoard


class GameState:
    """Class to manage the game state."""

    def __init__(self, player_dumb_X: RandomHandler, player_dumb_O: RandomHandler, player_smart_X: MinimaxHandler, player_smart_O: MinimaxHandler, last_move_naive : int, index: int, file_name: str):
        """
        Randomly simulates a game until last_move_naive is reached, upon which it switches to a smart player.
        Gamestates after last_move_naive are written to file.
        The outcome of the game is also written to file.
        
        :param playerX: Random actor for player X.
        :param playerO: Random actor for player O.
        """
        self.board = UTTTBoard()
        self.player_dumb_X = player_dumb_X
        self.player_dumb_O = player_dumb_O
        self.player_smart_X = player_smart_X
        self.player_smart_O = player_smart_O
        self.current_player: Player = "X"
        self.game_over: bool = False
        self.round_count: int = 0
        self.last_move_naive: int = last_move_naive
        self.boards_for_writing: List[Tuple[UTTTBoard, float]] = []
        self.index = index
        self.file_name = file_name

        # Current board to play in, None is any board
        self.current_forced_board_index: BoardIndex | None = None

    def run_game_naive(self) -> None:
        while not self.game_over and not self.round_count > self.last_move_naive:
            board_index, cell_index = (
                self.player_dumb_X.get_move(self.board, self.current_forced_board_index)
                if self.current_player == "X"
                else self.player_dumb_O.get_move(self.board, self.current_forced_board_index)
            )

            assert 0 <= board_index <= 8, "Board index must be between 0 and 8."
            assert 0 <= cell_index <= 8, "Cell index must be between 0 and 8."

            if (
                self.current_forced_board_index is not None
                and board_index != self.current_forced_board_index
            ):
                continue

            current_small_board = self.board.get_small_board(board_index)
            if current_small_board.winner is not None:
                continue

            if current_small_board.get_cell_value(cell_index) is not None:  # type: ignore
                continue

            self.round_count += 1

            self.board.make_move(board_index, cell_index, self.current_player)  # type: ignore

            # Allow the next move to be played only in the small board
            # corresponding to the cell index of the last move
            next_small_board = self.board.get_small_board(cell_index)
            if next_small_board.winner is None:
                self.current_forced_board_index = cell_index
            else:
                # If the next small board is already won, we can play anywhere
                # Set current_board_index to None to allow any board
                self.current_forced_board_index = None

            if self.board.winner is not None:
                self.game_over = True
                break

            self.__switch_player()


    def run_game_informed(self) -> None:
        while not self.game_over:
            move_maker = self.player_smart_X if self.current_player == "X" else self.player_smart_O
            [board_index, cell_index], confidence = (
                move_maker.get_move_confidence(self.board, self.current_forced_board_index)
            )

            assert 0 <= board_index <= 8, "Board index must be between 0 and 8."
            assert 0 <= cell_index <= 8, "Cell index must be between 0 and 8."

            if (
                self.current_forced_board_index is not None
                and board_index != self.current_forced_board_index
            ):
                continue

            current_small_board = self.board.get_small_board(board_index)
            if current_small_board.winner is not None:
                continue

            if current_small_board.get_cell_value(cell_index) is not None:  # type: ignore
                continue

            self.boards_for_writing.append((self.board.copy(), confidence))

            self.round_count += 1
            self.board.make_move(board_index, cell_index, self.current_player)  # type: ignore

            # Allow the next move to be played only in the small board
            # corresponding to the cell index of the last move
            next_small_board = self.board.get_small_board(cell_index)
            if next_small_board.winner is None:
                self.current_forced_board_index = cell_index
            else:
                # If the next small board is already won, we can play anywhere
                # Set current_board_index to None to allow any board
                self.current_forced_board_index = None
            

            if self.board.winner is not None:
                self.game_over = True
                break

            self.__switch_player()

    def run_game_praktikum(self) -> None:
        """
        Runs the game with both players, first with a naive player and then with an informed player,
        writng all boards to a file after the game is over.
        """
        self.run_game_naive()
        self.run_game_informed()
        self.write_boards_to_file()

    def __switch_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"

    def write_boards_to_file(self) -> None:
        with open(self.file_name, "a") as f:
            print(f"Writing {len(self.boards_for_writing)} boards to file {self.file_name} for index {self.index}")
            for board in self.boards_for_writing:
                f.write(f"{self.index};{board[0].to_csv_row()};{self.current_forced_board_index};{self.board.winner};{board[1]}\n") # stimmt der forced-board-index hier?