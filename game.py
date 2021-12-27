import random
import copy

class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def succ(self, state, piece, drop_phase): 
        if drop_phase:
            successors = []
            for row in range(len(state)):
                for col in range(len(state[0])):
                    if(state[row][col] == ' '):
                        successors.append((row,col))
        else:
            poss_moves = [-1,0,1] # Used https://stackoverflow.com/questions/2035522/get-adjacent-elements-in-a-two-dimensional-array for adjacent elements in array
            successors = []
            for row in range(len(state)):
                for col in range(len(state[0])):
                    if(state[row][col] == piece):
                        for r in poss_moves:
                            for c in poss_moves:
                                if(0 <= row + r < len(state[0]) and 0 <= col + c < len(state[0]) and state[row + r][col + c] == ' '):
                                    successors.append([(row + r, col + c), (row, col)])
        random.shuffle(successors)
        return successors

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        piece_count = 0
        for row in state:
            piece_count += row.count('b') + row.count('r')
        drop_phase = True if piece_count < 8 else False
        move = []
        alpha= -100000
        beta = 100000
        # ensure the destination (row,col) tuple is at the beginning of the move list
        if not drop_phase:
            successors = self.succ(state, self.my_piece, drop_phase)
            move = [(0,0),(0,0)]
            for successor in successors:
                tmp_state = copy.deepcopy(state)
                tmp_state[successor[1][0]][successor[1][1]] = ' '
                tmp_state[successor[0][0]][successor[0][1]] = self.my_piece 
                minimax = self.min_value(tmp_state, 0, alpha, beta, drop_phase)
                if(alpha < minimax):
                    move = successor
                    alpha = minimax
            return move

        successors = self.succ(state, self.my_piece, drop_phase)
        dest = [0,0]
        for successor in successors:
            row = successor[0]
            col = successor[1]
            tmp_state = copy.deepcopy(state)
            tmp_state[row][col] = self.my_piece
            minimax = self.min_value(tmp_state, 0, alpha, beta, drop_phase)
            if(alpha <= minimax):
                dest = (row,col)
                alpha = minimax
        move.insert(0, dest)
        return move

    def max_value(self, state, depth, alpha, beta, drop_phase):
        if(self.game_value(state) != 0):
            return self.game_value(state)
        if(depth >= 1):
            return self.heuristic_game_value(state)
        if(drop_phase):
            successors = self.succ(state, self.my_piece, drop_phase)
            for row,col in successors:
                state_copy = copy.deepcopy(state)
                state_copy[row][col] = self.my_piece
                alpha = max(alpha, self.min_value(state_copy, depth + 1, alpha, beta, drop_phase))
        else:
            successors = self.succ(state,self.my_piece, drop_phase)
            for successor in successors:
                state_copy = copy.deepcopy(state)
                state_copy[successor[1][0]][successor[1][1]] = ' ' # logic from place_piece
                state_copy[successor[0][0]][successor[0][1]] = self.my_piece
                alpha = max(alpha, self.min_value(state_copy, depth + 1, alpha, beta, drop_phase))
        return alpha
    
    def min_value(self, state, depth, alpha, beta, drop_phase):
        if(self.game_value(state) != 0):
            return self.game_value(state)
        if(depth >= 1):
            return self.heuristic_game_value(state)
        if(drop_phase):
            successors = self.succ(state, self.opp, drop_phase)
            for row,col in successors:
                state_copy = copy.deepcopy(state)
                state_copy[row][col] = self.opp
                beta = min(beta, self.max_value(state_copy, depth + 1, alpha, beta, drop_phase))
        else:
            successors = self.succ(state, self.opp, drop_phase)
            for successor in successors:
                state_copy = copy.deepcopy(state)
                state_copy[successor[1][0]][successor[1][1]] = ' '
                state_copy[successor[0][0]][successor[0][1]] = self.opp
                beta = min(beta, self.max_value(state_copy, depth + 1, alpha, beta, drop_phase))
        return beta
    
     # Used https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python/ for heuristic_game_value implementation with max, min vals
    def heuristic_game_value(self, state):
        terminal_val = self.game_value(state)
        if(terminal_val != 0):
            return terminal_val
        max_value = -2 # worse than worst case 
        min_val = 2 # worse than worst case
        states = []
        # horizontal
        for row in state:
            for col in range(2):
                for i in range(4):
                    states.append(row[col + i])
                max_value = max(max_value, states.count(self.my_piece) / 5) # dividing by 5,-5 to keep it close to -1,1 game value ranges (for all cases below)
                # inspiration from piazza post @629
                min_val = min(min_val, states.count(self.opp) / -5)
        states.clear()

        # vertical
        for col in range(5):
            for row in range(2):
                for i in range(4):
                    states.append(state[row + i][col])
                max_value = max(max_value, states.count(self.my_piece) / 5)
                min_val = min(min_val, states.count(self.opp) / -5)
        states.clear()

        # \ diagonal
        for row in range(2):
            for col in range(2):
                for i in range(4):
                    if col+i < 5 and row+i < 5:
                        states.append(state[row + i][col + i])
                max_value = max(max_value, states.count(self.my_piece) / 5)
                min_val = min(min_val, states.count(self.opp) / -5)
        states.clear()
            
        # / diagonal
        for row in range(2):
            for col in range(3,5):
                for i in range(4):
                    if col-i >= 0 and row+i < 5:
                        states.append(state[row + i][col - i])
                max_value = max(max_value, states.count(self.my_piece) / 5)
                min_val = min(min_val, states.count(self.opp) / -5)
        states.clear()
                
        # 3x3 square
        for row in range(1, 4):
            for col in range(1, 4):
                states.append(state[row - 1][col - 1])
                states.append(state[row - 1][col + 1])
                states.append(state[row + 1][col - 1])
                states.append(state[row + 1][col + 1])
                max_value = max(max_value, states.count(self.my_piece) / 5)
                min_val = min(min_val, states.count(self.opp) / -5)
        states.clear()
        return max_value + min_val

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and 3x3 square corners wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == state[row+3][col+3]:
                    return 1 if state[row][col]==self.my_piece else -1
            
        # check / diagonal wins
        for row in range(2):
            for col in range(3,5):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col-1] == state[row+2][col-2] == state[row+3][col-3]:
                    return 1 if state[row][col]==self.my_piece else -1
        # check 3x3 square wins
        for row in range(4):
            for col in range(4):
                if state[row][col] == ' ' and state[row - 1][col - 1] != ' ' and state[row - 1][col - 1] == state[row - 1][col + 1] == state[row + 1][col - 1] == state[row + 1][col + 1]:
                    return 1 if state[row - 1][col - 1]==self.my_piece else -1
        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
