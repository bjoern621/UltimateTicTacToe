from game_state import GameState
from move_handlers.console_handler import ConsoleHandler
from uttt_board import UTTTBoard


def main():
    print("Running...")

    board = UTTTBoard()

    playerX = ConsoleHandler("X")
    playerO = ConsoleHandler("O")

    state = GameState(board, playerX, playerO)

    state.run_game_loop()


if __name__ == "__main__":
    main()
