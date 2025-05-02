from move_handlers.mcts_tree import Move, SearchTreeNode, State
from move_handlers.move_handler import MoveHandler
from ttt_board import CellIndex, Player
from uttt_board import BoardIndex, UTTTBoard
import threading

class AIHandler(MoveHandler):
    def __init__(self, player: Player, max_time: int):
        self.player = player
        self.max_time = max_time
        self.search_tree: SearchTreeNode | None = None

    def get_move(self, board: UTTTBoard, forced_board: BoardIndex | None) -> tuple[BoardIndex, CellIndex]:
        last_move = Move(forced_board, 0) if forced_board is not None else None
        curr_state = State(last_move, self.player, board, forced_board)
        
        # If a search tree already exists, reuse it
        if self.search_tree is not None:
            # Find the child node corresponding to the last move
            matching_child = None
            for child in self.search_tree.children:
                if child.state.last_move == last_move:
                    matching_child = child
                    break

            if matching_child:
                # Reuse the subtree
                self.search_tree = matching_child
                self.search_tree.parent = None  # Detach from the old parent
            else:
                # If no matching child is found, rebuild the tree
                self.search_tree = SearchTreeNode(curr_state, None)
        else:
            # Initialize the search tree for the first move
            self.search_tree = SearchTreeNode(curr_state, None)
        
        # Prepare simulation worker
        stop_event = threading.Event()
        sim_thread = threading.Thread(target=self.run_simulation, args=(self.search_tree, stop_event))
        
        # Run simulation for set ammount of time
        sim_thread.start()
        sim_thread.join(timeout=self.max_time)
        if (sim_thread.is_alive()):
            stop_event.set()
            sim_thread.join()

        move = self.select_move(self.search_tree)

        assert move[0] == forced_board if forced_board is not None\
            else board.get_small_board(move[0]).winner is None,\
            "Fatal error: AIHandler selected a move to an invalid board."
        
        assert board.get_small_board(move[0]).get_cell_value(move[1]) is None,\
            "Fatal error: AIHandler selected a cell that is already claimed."

        return move
    
    def run_simulation(self, search_tree: SearchTreeNode, stop_event: threading.Event):
        while not stop_event.is_set():
            search_tree.simulate_game()

    def select_move(self, search_tree: SearchTreeNode) -> tuple[BoardIndex, CellIndex]:
        assert search_tree.children, "Fatal error: No children have been calculated yet."

        #for child in search_tree.children:
        #    print(f"Child node: total_runs={child.total_runs}, wins={child.wins}, losses={child.losses}")

        max_runs = max(child.total_runs for child in search_tree.children)
        candidates = [child for child in search_tree.children if child.total_runs == max_runs]
        best_move = max(candidates, key=lambda child: child.wins)
        
        print(f"""MTCS had time to think. Best Node: 
              {best_move.total_runs} Runs, 
              {best_move.wins} Wins, 
              {best_move.losses} Losses.
              """)        

        # Update the search tree to the best move for the next turn
        self.search_tree = best_move
        self.search_tree.parent = None

        return best_move.state.last_move.board, best_move.state.last_move.cell