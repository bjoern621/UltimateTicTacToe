from ttt_board import CellIndex, Player, Winner
from uttt_board import BoardIndex, UTTTBoard
from typing import List, Optional, cast
import random
import time
import numpy as np

class Move:
    def __init__(self, board: BoardIndex, cell: CellIndex):
        self.board: BoardIndex = board
        self.cell: CellIndex = cell

class State:
    """The current state of the board, including the move that lead to it"""
    def __init__(self, last_move: Move | None, turn: Player, board: UTTTBoard, forced_board: BoardIndex | None):
        self.last_move: Move | None = last_move # Optional for the root node
        self.turn: Player = turn
        self.board: UTTTBoard = board
        self.forced_board: BoardIndex | None = forced_board
    
    def get_possible_moves(self) -> List[Move]:
        """Returns a list of all legal moves from the current state"""
        possible_moves: List[Move] = []
        possible_boards: List[BoardIndex] = []
        if self.forced_board is None:
            possible_boards = [
                cast(BoardIndex, board_index_int) for board_index_int in range(9)
                if self.board.get_small_board(cast(BoardIndex, board_index_int)).winner is None
            ]
        else: 
            possible_boards = [self.forced_board]
        
        for board_index in possible_boards:
            small_board = self.board.get_small_board(board_index)
            for cell_index_int in range(9):
                cell_index = cast(CellIndex, cell_index_int)
                if small_board.get_cell_value(cell_index) is None:
                    possible_moves.append(Move(board_index, cell_index))

        return possible_moves

class SearchTreeNode:
    """A node in the search tree"""
    def __init__(self, state: State, parent: Optional['SearchTreeNode']):
        self.state: State = state
        self.parent: SearchTreeNode | None = parent
        self.children: List[SearchTreeNode] = []

        # List of possible moves from this state
        self.open_moves: List[Move] = self.state.get_possible_moves()

        # Stats
        self.wins: float = 0
        self.total_runs: int = 0

        # Timing stats (profiling only)
        self.selection_time: float = 0
        self.expansion_time: float = 0
        self.simulation_time: float = 0
        self.backpropagation_time: float = 0

    def simulate_game(self):
        """Executes one iteration of the MCTS algorithm"""
        # Selection - traverse tree until leaf node is reached
        selection_start = time.perf_counter_ns()
        leaf = self.__choose_child()
        self.selection_time = time.perf_counter_ns() - selection_start

        # Expansion - if node is not terminal and not expanded, expand it
        expansion_start = time.perf_counter_ns()
        if leaf.state.board.winner is None:
            assert leaf.open_moves, "No moves available to expand"
            leaf = leaf.__expand_child()
        else :
            # Skip expansion and simulation if the game is over
            leaf.__back_propagate(leaf.state.board.winner)
            return
        self.expansion_time = time.perf_counter_ns() - expansion_start

        # Simulation - play out the game from the new node
        simulation_start = time.perf_counter_ns()
        winner = leaf.__run_simulation()
        self.simulation_time = time.perf_counter_ns() - simulation_start
        
        # Backpropagation - update statistics up the tree
        backpropagation_start = time.perf_counter_ns()
        leaf.__back_propagate(winner)
        self.backpropagation_time = time.perf_counter_ns() - backpropagation_start

    def __choose_child(self) -> 'SearchTreeNode':
        """
        Selection Step: Traverse the tree using UCT until reaching a leaf node
        """
        current_node = self
        while current_node.children:
            # If there are unexplored moves, return
            # this node to expand it
            if current_node.open_moves:
                return current_node
            
            # Otherwise use UCT to select the best child
            current_node = max(
                current_node.children,
                key=lambda child: child.__calc_potential()
            )
        
        return current_node
    
    def __expand_child(self) -> 'SearchTreeNode':
        """
        Expands the current node by adding a child node for the
        move that was just made.\\
        Returns the new child node.\\
        Expansion-Step
        """
        assert self.open_moves, "No moves available to expand"

        # Select a random move from the available moves for expansion
        move = self.open_moves.pop(random.randrange(len(self.open_moves)))

        new_board = self.state.board.copy()
        new_board.make_move(move.board, move.cell, self.state.turn)
        forced_board = move.cell if new_board.get_small_board(move.cell).winner is None else None
        new_state = State(move, "O" if self.state.turn == "X" else "X", new_board, forced_board)
        new_node = SearchTreeNode(new_state, self)
        self.children.append(new_node)

        return new_node

    def __run_simulation(self) -> Winner:
        """
        Simulates an entire run from a provided state till
        the game ends.\\
        Simulation-Step
        """
        sim_state = State(
            self.state.last_move, 
            self.state.turn, 
            self.state.board.copy(), 
            self.state.forced_board
        )
        sim_node = SearchTreeNode(sim_state, self)

        while sim_state.board.winner is None:
            # Pick a random move from the available moves
            moves = sim_node.open_moves
            move = moves[random.randrange(len(moves))]
            sim_state.board.make_move(move.board, move.cell, sim_state.turn)

            # Prepare next turn
            sim_state.turn = "O" if sim_state.turn == "X" else "X"
            sim_state.forced_board = move.cell\
                if sim_state.board.get_small_board(move.cell).winner is None\
                else None
            sim_node.open_moves = sim_state.get_possible_moves()

        return sim_state.board.winner
    
    def __back_propagate(self, winner: Winner):
        """
        Updates the stats of all nodes in the simulated
        path.\\
        Backpropagation-Step
        """
        # Determine the player who made the move leading to this node's state
        move_maker = "O" if self.state.turn == "X" else "X"
        
        if winner == move_maker:
            self.wins += 1
        elif winner == "Draw":
            self.wins += 0.5

        self.total_runs += 1

        if self.parent is not None:
            self.parent.__back_propagate(winner)
    
    def print_tree(self, level: int = 0):
        """
        Prints the search tree for debugging purposes
        """
        indent = " " * (level * 4)
        print(f"{indent}Node: Move: {"Tree" if self.state.last_move is None else [self.state.last_move.board, self.state.last_move.cell]}, Wins: {self.wins}, Total Runs: {self.total_runs}")
        if level > 2 and self.children:
            print(f"{indent + "    "}Children: {len(self.children)}, Open Moves: {len(self.open_moves)}, Total Wins: {sum(child.wins for child in self.children)}, Total Runs: {sum(child.total_runs for child in self.children)}")
            return # Limit the depth of the tree to print
        for child in self.children:
            child.print_tree(level + 1)

    def __calc_potential(self) -> float:
        """
        Calculates the potential of a node based on the 
        [UCB1 strategy](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation)
        """
        assert self.parent is not None, "Cannot calculate potential for root node"

        if self.total_runs == 0:
            return np.inf
        w = self.wins
        n_i = self.total_runs
        n = self.parent.total_runs
        return w / n_i + np.sqrt((2 * np.log(n)) / n_i)