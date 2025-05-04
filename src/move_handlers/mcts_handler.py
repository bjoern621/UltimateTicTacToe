from move_handlers.mcts_tree import SearchTreeNode, State
from move_handlers.move_handler import MoveHandler
from ttt_board import CellIndex, Player
from uttt_board import BoardIndex, UTTTBoard
import threading

class MCTSHandler(MoveHandler):
    """Monte Carlo Tree Search (MCTS) move handler."""
    __name__ = "MCTS"

    def __init__(self, player: Player, max_time: float, log: bool = True):
        super().__init__(player, log)
        self.max_time: float = max_time
        self.search_tree: SearchTreeNode | None = None
        self.__name__ = f"MCTS (Time: {max_time})"
        self.total_simulations = 0
        self.total_iterations = 0

    def get_max_value(self) -> float:
        return self.total_simulations / self.total_iterations if self.total_iterations > 0 else 0

    def get_move(self, board: UTTTBoard, forced_board: BoardIndex | None) -> tuple[BoardIndex, CellIndex]:
        # Readjust search tree for the opponent's move
        self.search_tree = self.get_new_root(board, forced_board)
        
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
    
    def get_new_root(self, board: UTTTBoard, forced_board: BoardIndex | None) -> SearchTreeNode:
        """
        Adjusts the search tree to account for the opponent's move.\\
        If the move is already in the search tree, it will be selected as the new root node.\\
        If the move is not in the search tree, a new search tree node will be created.\\
        If the search tree is empty, a new search tree node will be created.
        """
        curr_state = State(None, self.player, board, forced_board)
        if self.search_tree is None:
            return SearchTreeNode(curr_state, None)
        
        board_hash = board.get_hashable_state()

        for child in self.search_tree.children:
            if child.state.board.get_hashable_state() == board_hash and child.state.forced_board == forced_board:
                child.parent = None  # Detach from the old parent
                return child
        
        return SearchTreeNode(curr_state, None)
    
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

        if self.log: 
            print(f"Selection time per iteration:       {time_selection/search_tree.total_runs} ns")
            print(f"Expansion time per iteration:       {time_expansion/search_tree.total_runs} ns")
            print(f"Simulation time per iteration:      {time_simulation/search_tree.total_runs} ns")
            print(f"Backpropagation time per iteration: {time_backpropagation/search_tree.total_runs} ns")

        self.total_simulations += search_tree.total_runs
        self.total_iterations += 1

    def select_move(self, search_tree: SearchTreeNode) -> tuple[BoardIndex, CellIndex]:
        assert search_tree.children, "Fatal error: No children have been calculated yet."

        max_runs = max(child.total_runs for child in search_tree.children)
        candidates = [child for child in search_tree.children if child.total_runs == max_runs]
        best_move = max(candidates, key=lambda child: child.wins)

        if self.log and best_move.state.last_move: print(f"""MTCS had time to think. Best Node: {best_move.state.last_move.board}, {best_move.state.last_move.cell}
              {best_move.total_runs} Runs, 
              {best_move.wins} Wins.
              Total Runs: {search_tree.total_runs}
              """)
            
        with open("mcts_runtime.csv", "a") as f:
            f.write(f"{self.max_time};{search_tree.total_runs};{best_move.wins/best_move.total_runs}\n")
        
        # Update the search tree to the best move for the next turn
        self.search_tree = best_move
        self.search_tree.parent = None

        assert best_move.state.last_move is not None, "Fatal error: Best move is None."

        return best_move.state.last_move.board, best_move.state.last_move.cell