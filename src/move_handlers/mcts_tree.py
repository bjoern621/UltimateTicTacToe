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
    def __init__(self, last_move: Move | None, turn: Player, board: UTTTBoard, forced_board: BoardIndex):
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

        # Stats
        self.wins: float = 0
        self.total_runs: int = 0

        # Analyse
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
        if leaf.state.board.winner is None and not leaf.children:
            leaf.__expand_children()
            # Select a child node randomly for simulation
            if leaf.children:
                leaf = leaf.children[random.randrange(len(leaf.children))]
        self.expansion_time = time.perf_counter_ns() - expansion_start

        # Simulation - play out the game from this position
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
            # If there are unexplored children, pick one randomly
            unexplored = [child for child in current_node.children if child.total_runs == 0]
            if unexplored:
                return random.choice(unexplored)
            
            # Otherwise use UCT to select the best child
            current_node = max(
                current_node.children,
                key=lambda child: child.__calc_potential()
            )
        
        return current_node

    def __expand_children(self):
        """
        Calculates all moves available from the current state
        and creates a child node for each move.\\
        Expansion-Step
        """
        if self.state.board.winner is None:
            self.children = self.__get_children()
            assert self.children, "No children found despite the game not being over"

    def __get_children(self) -> List['SearchTreeNode']:
        """
        Calculates all moves available from the current state
        """
        possible_moves = self.state.get_possible_moves()
        possible_children = []
        turn = self.state.turn
        next_turn = "O" if turn == "X" else "X"
        
        for move in possible_moves:
            # Create a new board for each child
            sim_board = self.state.board.copy()

            assert sim_board.get_small_board(move.board)\
                            .get_cell_value(move.cell) is None,\
                            "Invalid move passed to selection-step"
            
            sim_board.make_move(move.board, move.cell, turn)

            forced_board = move.cell if sim_board.get_small_board(move.cell).winner is None else None

            new_state = State(move, next_turn, sim_board, forced_board)
            possible_children.append(SearchTreeNode(new_state, self))

        return possible_children

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

        while sim_state.board.winner is None:
            # Pick a random move from the available moves
            moves = sim_state.get_possible_moves()
            move = moves[random.randrange(len(moves))]
            sim_state.board.make_move(move.board, move.cell, sim_state.turn)

            # Prepare next turn
            sim_state.turn = "O" if sim_state.turn == "X" else "X"
            sim_state.forced_board = move.cell\
                if sim_state.board.get_small_board(move.cell).winner is None\
                else None

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
        print(f"{indent}Node: {None if self.state.last_move is None else [self.state.last_move.board, self.state.last_move.cell]}, Wins: {self.wins}, Total Runs: {self.total_runs}")
        for child in self.children:
            child.print_tree(level + 1)

        return

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