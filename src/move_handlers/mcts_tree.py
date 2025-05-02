from ttt_board import CellIndex, Player, Winner
from uttt_board import BoardIndex, UTTTBoard
from typing import List, Optional, cast
import random
import copy
import math

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
        self.wins: int = 0
        self.losses: int = 0
        self.total_runs: int = 0

    def simulate_game(self):
        self.__choose_child()

    def __choose_child(self):
        """
        Traverses the search tree until it reaches an
        unexplored node, on which the next simulation 
        will be run.\\
        Selection-Step
        """
        if not self.children:
            self.__expand_children()
        else:
            # Traverse Search Tree from the current game state
            # until leaf node is reached
            current_node = self
            while current_node.children:
                unexplored_children = [child for child in current_node.children if child.total_runs == 0]
                if unexplored_children:
                    # If there are unexplored children, choose one of them randomly
                    # Skips the calculation of potential (+inf)
                    current_node = random.choice(unexplored_children)
                    break
                # Otherwise, choose the child with the highest potential
                current_node = max(
                    current_node.children, 
                    key=lambda child: self.__calc_potential(child)
                )
            # Expand the leaf node
            current_node.__expand_children()
    
    def __expand_children(self):
        """
        Calculates all moves available from the current state
        and creates a child node for each move.\\
        Expansion-Step
        """
        if self.state.board.winner is not None: 
            self.__back_propagate(self.state.board.winner)
        else:
            self.children = self.__get_children(self.state)

            assert self.children, "No children found despite the game not being over"
            
            selected_child = random.choice(self.children)
            selected_child.__run_simulation()

    def __get_children(self, state: State) -> List['SearchTreeNode']:
        """
        Calculates all moves available from the current state
        """
        possible_moves = state.get_possible_moves()
        possible_children = []
        turn = state.turn
        next_turn = "O" if turn == "X" else "X"
        
        for move in possible_moves:
            sim_board = copy.deepcopy(state.board) # TODO: Checken, ob das nötig ist
            sim_board.make_move(move.board, move.cell, turn)

            forced_board = move.cell if sim_board.get_small_board(move.cell).winner is None else None

            new_state = State(move, next_turn, sim_board, forced_board)
            possible_children.append(SearchTreeNode(new_state, self))

        return possible_children

    def __run_simulation(self):
        """
        Simulates an entire run from a provided state till
        the game ends.\\
        Simulation-Step
        """
        #print(f"Simulating game for node with state: {self.state}")
        sim_state = copy.deepcopy(self.state)

        while sim_state.board.winner is None:
            # Make random move
            move = random.choice(sim_state.get_possible_moves())
            sim_state.board.make_move(move.board, move.cell, sim_state.turn)

            # Prepare next turn
            sim_state.turn = "O" if sim_state.turn == "X" else "X"
            sim_state.forced_board = move.cell\
                if sim_state.board.get_small_board(move.cell).winner is None\
                else None
        
        self.__back_propagate(sim_state.board.winner)
    
    def __back_propagate(self, winner: Winner):
        """
        Updates the stats of all nodes in the simulated
        path.\\
        Backpropagation-Step
        """
        #print(f"Backpropagating. Node total_runs: {self.total_runs}, wins: {self.wins}, losses: {self.losses}")
        if winner == self.state.turn:
            self.wins += 1
        elif winner == "Draw":
            self.wins += 0.5
            self.losses += 0.5
        else:
            self.losses += 1

        self.total_runs += 1

        if self.parent is not None:
            self.parent.__back_propagate(winner)
        else:
            print(f"{winner} won the game!")
            self.print_tree()
    
    def print_tree(self, level: int = 0):
        """
        Prints the search tree for debugging purposes
        """
        indent = " " * (level * 4)
        print(f"{indent}Node: {None if self.state.last_move is None else [self.state.last_move.board, self.state.last_move.cell]}, Wins: {self.wins}, Losses: {self.losses}, Total Runs: {self.total_runs}")
        for child in self.children:
            child.print_tree(level + 1)

        return

    def __calc_potential(self, node: 'SearchTreeNode') -> float:
        """
        Calculates the potential of a node based on the 
        [UCB1 strategy](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation)
        """
        assert node.parent is not None, "Root node was passed to __calc_potential"

        if node.total_runs == 0:
            return math.inf
        w = node.wins
        n = node.total_runs
        c = math.sqrt(2)
        t = node.parent.total_runs
        return w / n + c * math.sqrt(math.log(t) / n)