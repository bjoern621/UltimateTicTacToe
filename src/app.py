from game_state import GameState
from uttt_board import UTTTBoard


def main():
    print("Running...")

    board = UTTTBoard()

    # Example: Make a move in the top-left small board (index 0), at its center cell (index 4)
    # ultimate_game.get_small_board(0).board[4] = 'X'
    # ultimate_game.next_board_index = 4 # Next move must be in board 4

    state = GameState(board)

    state.run_game_loop()

    # You would add game loop logic here


if __name__ == "__main__":
    main()
