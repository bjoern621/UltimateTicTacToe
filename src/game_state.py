from move_handlers.minimax_handler import MinimaxHandler
from move_handlers.move_handler import MoveHandler
from move_handlers.random_handler import RandomHandler
from ttt_board import Player
from uttt_board import BoardIndex, UTTTBoard


class GameState:
    """Class to manage the game state."""

    def __init__(self, player_dumb: RandomHandler, player_smart: MinimaxHandler, last_move_naive : int):
        """
        Randomly simulates a game until last_move_naive is reached, upon which it switches to a smart player.
        Gamestates after last_move_naive are written to file.
        The outcome of the game is also written to file.
        
        :param playerX: Random actor for player X.
        :param playerO: Random actor for player O.
        """
        self.board = UTTTBoard()
        self.player_dumb = player_dumb
        self.player_smart = player_smart
        self.current_player: Player = "X"
        self.game_over: bool = False
        self.round_count: int = 0
        self.last_move_naive: int = last_move_naive

        # Current board to play in, None is any board
        self.current_forced_board_index: BoardIndex | None = None

    def run_game_naive(self) -> None:
        while not self.game_over and not self.round_count > self.last_move_naive:
            
            board_index, cell_index = (
                self.player_dumb.get_move(self.board, self.current_forced_board_index)
                if self.current_player == "X"
                else self.player_dumb.get_move(self.board, self.current_forced_board_index)
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
        # Hier in Datei schreiben

    def run_game_informed(self) -> None:
        while not self.game_over and not self.round_count > self.last_move_naive:
            board_index, cell_index = (
                self.player_smart.get_move(self.board, self.current_forced_board_index)
                if self.current_player == "X"
                else self.player_smart.get_move(self.board, self.current_forced_board_index)
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
            
            # Hier in Datei schreiben

            if self.board.winner is not None:
                self.game_over = True
                break


            self.__switch_player()
        
        # Hier in allen bisherigen Zeilen den Winner eintragen


    def run_game_praktikum(self) -> None:
        self.run_game_naive()
        self.run_game_informed()

    # def __init__(self, board: UTTTBoard, playerX: MoveHandler, playerO: MoveHandler):
    #     self.board = board
    #     self.playerX = playerX
    #     self.playerO = playerO
    #     self.current_player: Player = "X"
    #     self.game_over: bool = False

    #     # Current board to play in, None is any board
    #     self.current_forced_board_index: BoardIndex | None = None

    def __switch_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"

    def run_game_loop(self, log: bool = True) -> None:
        """Main game loop."""

        while not self.game_over:
            if log:
                print("/\\" * 40)

                self.board.display_board(self.current_forced_board_index)

            board_index, cell_index = (
                self.playerX.get_move(self.board, self.current_forced_board_index)
                if self.current_player == "X"
                else self.playerO.get_move(self.board, self.current_forced_board_index)
            )

            assert 0 <= board_index <= 8, "Board index must be between 0 and 8."
            assert 0 <= cell_index <= 8, "Cell index must be between 0 and 8."

            if (
                self.current_forced_board_index is not None
                and board_index != self.current_forced_board_index
            ):
                if log: print(
                    f"Invalid move! You must play in board {self.current_forced_board_index}."
                )
                continue

            current_small_board = self.board.get_small_board(board_index)
            if current_small_board.winner is not None:
                if current_small_board.winner == "Draw":
                    if log: print(f"Board {board_index} is a draw! Try again.")
                else:
                    if log: print(
                        f"Board {board_index} already won by {current_small_board.winner}! Try again."
                    )
                continue

            if current_small_board.get_cell_value(cell_index) is not None:  # type: ignore
                if log: print("Cell already occupied! Try again.")
                continue

            self.board.make_move(board_index, cell_index, self.current_player)  # type: ignore

            if self.board.winner is not None:
                self.game_over = True
                if self.board.winner == "Draw":
                    if log: print("Game over! It's a draw!")
                else:
                    if log: print(f"Game over! {self.board.winner} wins!")
                if log: self.board.display_board(None)
                break

            # Allow the next move to be played only in the small board
            # corresponding to the cell index of the last move
            next_small_board = self.board.get_small_board(cell_index)
            if next_small_board.winner is None:
                self.current_forced_board_index = cell_index
            else:
                # If the next small board is already won, we can play anywhere
                # Set current_board_index to None to allow any board
                self.current_forced_board_index = None

            self.__switch_player()
