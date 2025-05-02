from move_handlers.mcts_tree import Move, SearchTreeNode, State
from move_handlers.move_handler import MoveHandler
from ttt_board import CellIndex, Player
from uttt_board import BoardIndex, UTTTBoard
import threading

class MCTSHandler(MoveHandler):
    def __init__(self, player: Player, max_time: int):
        self.player = player
        self.max_time = max_time
        self.search_tree: SearchTreeNode | None = None

    def get_move(self, board: UTTTBoard, forced_board: BoardIndex | None) -> tuple[BoardIndex, CellIndex]:
        board_hash = board.get_hashable_state()
        curr_state = State(None, self.player, board, forced_board)

        if self.search_tree is None:
            self.search_tree = SearchTreeNode(curr_state, None)
        
        matching_child = None
        for child in self.search_tree.children:
            if child.state.board.get_hashable_state() == board_hash:
                matching_child = child
                break
        
        if matching_child:
            self.search_tree = matching_child
            self.search_tree.parent = None  # Detach from the old parent
        else:
            # If no matching child is found, create a new search tree node
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
        time_selection = 0
        time_expansion = 0
        time_simulation = 0
        time_backpropagation = 0
        while not stop_event.is_set():
            search_tree.simulate_game()
            
            time_selection += search_tree.selection_time
            time_expansion += search_tree.expansion_time
            time_simulation += search_tree.simulation_time
            time_backpropagation += search_tree.backpropagation_time

        print(f"Selection time per iteration:       {time_selection/search_tree.total_runs} ns")
        print(f"Expansion time per iteration:       {time_expansion/search_tree.total_runs} ns")
        print(f"Simulation time per iteration:      {time_simulation/search_tree.total_runs} ns")
        print(f"Backpropagation time per iteration: {time_backpropagation/search_tree.total_runs} ns")

        # self.search_tree.print_tree()

    def select_move(self, search_tree: SearchTreeNode) -> tuple[BoardIndex, CellIndex]:
        assert search_tree.children, "Fatal error: No children have been calculated yet."

        max_runs = max(child.total_runs for child in search_tree.children)
        candidates = [child for child in search_tree.children if child.total_runs == max_runs]
        best_move = max(candidates, key=lambda child: child.wins)

        print(f"""MTCS had time to think. Best Node: 
              {best_move.total_runs} Runs, 
              {best_move.wins} Wins.
              Total Runs: {search_tree.total_runs}
              """)        

        # Update the search tree to the best move for the next turn
        self.search_tree = best_move
        self.search_tree.parent = None

        return best_move.state.last_move.board, best_move.state.last_move.cell