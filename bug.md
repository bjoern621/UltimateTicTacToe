BUG

```
(previous Cache size: 16098)
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
-- player O made a move --
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

 ╭───╮ ║ ╭───╮ ║ ╲   ╱
 │   │ ║ │   │ ║  ╲ ╱
 │   │ ║ │   │ ║   ╳
 │   │ ║ │   │ ║  ╱ ╲
 ╰───╯ ║ ╰───╯ ║ ╱   ╲
═══════╬═══════╬═══════
 ╲   ╱ ║ ╲   ╱ ║ O│X│O
  ╲ ╱  ║  ╲ ╱  ║ ─┼─┼─
   ╳   ║   ╳   ║  │X│O
  ╱ ╲  ║  ╱ ╲  ║ ─┼─┼─
 ╱   ╲ ║ ╱   ╲ ║ X│O│
═══════╬═══════╬═══════
 ╭───╮ ║ ╲   ╱ ║ ╲   ╱
 │   │ ║  ╲ ╱  ║  ╲ ╱
 │   │ ║   ╳   ║   ╳
 │   │ ║  ╱ ╲  ║  ╱ ╲
 ╰───╯ ║ ╱   ╲ ║ ╱   ╲


Iterative Minimax (X) thinking... Forced board: None
  Starting search at depth 1...
  Finished! Found move: (5, 8) with score: 320.0.
  Starting search at depth 2...
  Finished! Found move: None with score: 300.0.
  Starting search at depth 3...
  Finished! Found move: (5, 8) with score: 0.
  Starting search at depth 4...
  Finished! Found move: None with score: 0.
  Starting search at depth 5...
  Finished! Found move: None with score: 0.
  Starting search at depth 6...
  Finished! Found move: None with score: 0.
  Starting search at depth 7...
  Finished! Found move: None with score: 0.
  Starting search at depth 8...
  Finished! Found move: None with score: 0.
  Starting search at depth 9...
  Finished! Found move: None with score: 0.
  Starting search at depth 10...
  Finished! Found move: None with score: 0.
  Starting search at depth 11...
  Finished! Found move: None with score: 0.
  Starting search at depth 12...
  Finished! Found move: None with score: 0.
  Starting search at depth 13...
  Finished! Found move: (5, 8) with score: 0.
  Starting search at depth 14...
  Finished! Found move: (5, 3) with score: 0.
  Starting search at depth 15...
  Finished! Found move: None with score: 0.
  Starting search at depth 16...
  Finished! Found move: None with score: 0.
  Starting search at depth 17...
  Finished! Found move: (5, 8) with score: 0.
  Starting search at depth 18...
  Finished! Found move: None with score: 10000.
  Found guaranteed winning/losing move at depth 18.
Traceback (most recent call last):
  File "C:\Users\bless\git\UltimateTicTacToe\src\app.py", line 86, in <module>
    main()
  File "C:\Users\bless\git\UltimateTicTacToe\src\app.py", line 64, in main
    state.run_game_loop()
  File "C:\Users\bless\git\UltimateTicTacToe\src\game_state.py", line 31, in run_game_loop
    self.playerX.get_move(self.board, self.current_forced_board_index)
  File "C:\Users\bless\git\UltimateTicTacToe\src\move_handlers\minimax_iterative_handler.py", line 85, in get_move
    assert best_move is not None, "Minimax returned None for the move."
AssertionError: Minimax returned None for the move.
```

self.memo.clear()

seems to 'fix' the issue?
