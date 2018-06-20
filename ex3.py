import time
import random
import checker
import copy as c

ids = ["000000000", "111111111"]

def get_next_tile(current_tile,action):
    current_tile_x, current_tile_y = current_tile
    if action == "U":
        next_tile = (current_tile_x - 1, current_tile_y)
    if action == "R":
        next_tile = (current_tile_x, current_tile_y + 1)
    if action == "D":
        next_tile = (current_tile_x + 1, current_tile_y)
    if action == "L":
        next_tile = (current_tile_x, current_tile_y - 1)
    return next_tile

def cal_diff(loc1,loc2):
    return [loc1[0]-loc2[0],loc1[1]-loc2[1]]

def get_eligible_moves(tile):
    r,c=tile
    moves = []
    if r > 0:
        moves+=["D"]
    elif r < 0:
        moves += ["U"]
    if c > 0:
        moves+= ["R"]
    elif c < 0:
        moves+=["L"]
    return moves

def convert_dictionary_to_tuple(poss_state_dictionary):
    # TODO: Function receives a dictionary of dictionaries
    # Returns flat 5-tuple of states representing pacman, ghost actions
    # e.g.
    # poss_state_dictionary = {'U': {'green': ['U', 'L'], 'blue': ['D'],
    # 'yellow': ['L']}, 'R': {'green': ['L'], 'blue': ['D', 'R'], 'yellow': ['D']}, 'L': {'green': ['L'], 'blue': ['D', 'L'], 'yellow': ['D', 'L']}}
    #
    # returns: list of list of possible moves ordered by pacman, blue,yellow,green,red
    # 0 if doesn't exist
    # [["U","D","L","L",0],["U","D","L","L",0],["R"....etc.]]

    #
    return [["U","D","L","L",0]]
    pass

class PacmanState:

    def __init__(self,board_state,last_pacman_action=None,e1=0,e2=0,e3=0):
        # board
        self.state, self.special_things = checker.problem_to_state(board_state)
        self.last_pacman_action=last_pacman_action
        self.last_type_eaten = 0

        self.score = 0
        self.h_val = 0

        # Expected Values for each type of dot
        self.e1=e1
        self.e2=e2
        self.e3=e3

    def move_pacman_ghosts(self,actions_list):
        # actions_list is ordered by pacman, blue, yellow, green, red
        # 0 if no action taken / ghost doesn't exist
        # ["U","D","L","L",0]
        # means that pacman moves U, blue ghost moves D, green moves L, yellow moves L, red doesn't exist.
        print("Actions List")
        print(actions_list)
        # what is next_pacman tile ==
        next_tile = get_next_tile(self.special_things["pacman"],actions_list[0])

        self.last_type_eaten = self.state[next_tile]%10
        self.last_pacman_action = actions_list[0]

        # TODO Update state of board for pacman action and ghosts
        # TODO Update h_val with new state of board
        # If pacman dies, handle properly.

        pass

    def compute_h(self):
        # TODO
        # This function computes the heuristic value of the current state.
        # The function doesn't return anything.
        # for example, h_val = score - (# of ghosts remaining)^2

        self.h_val = 0 #new value
        pass

    def get_possible_moves(self,special_thing):
        # pacman is alive and cannot move into a wall or a ghost or a poison!
        special_thing_moves = []
        current_tile = self.special_things[special_thing]

        next_tile = get_next_tile(current_tile,"U")
        if self.state[next_tile] not in checker.BLOCKING_CODES:
            special_thing_moves += ["U"]

        next_tile = get_next_tile(current_tile, "R")
        if self.state[next_tile] not in checker.BLOCKING_CODES:
            special_thing_moves += ["R"]

        next_tile = get_next_tile(current_tile, "D")
        if self.state[next_tile] not in checker.BLOCKING_CODES:
            special_thing_moves += ["D"]

        next_tile = get_next_tile(current_tile, "L")
        if self.state[next_tile] not in checker.BLOCKING_CODES:
            special_thing_moves += ["L"]

        return special_thing_moves

    def get_ghost_moves(self,next_pacman_location):
        # TODO: Returns dictionary of all ghosts and their possible moves
        ghost_dict = {}

        #pacman_location = self.special_things["pacman"]
        for color in checker.COLOR_CODES:
            if color in self.special_things:
                # find possible moves for the special things
                ghost_location = self.special_things[color]
                location_diff = cal_diff(next_pacman_location,ghost_location)
                moves = get_eligible_moves(location_diff)
                ghost_dict[color] = moves

        return ghost_dict

    def get_next_moves(self):
        possible_state_diction = {}
        
        pacman_moves = self.get_possible_moves("pacman")

        for action in pacman_moves:

            # get Pacman's new location.
            current_pacman_location = self.special_things["pacman"]
            next_pacman_tile = get_next_tile(current_pacman_location, action)

            ghost_moves = self.get_ghost_moves(next_pacman_tile)
            possible_state_diction[action]=ghost_moves

        return possible_state_diction


class PacmanController:
    """This class is a controller for a pacman agent."""
    def __init__(self, state, steps):
        """Initialize controller for given the initial setting.
        This method MUST terminate within the specified timeout."""
        # print('COMPLETE init ')

        # Keep track of the old value
        self.last_state_reward = 0
        self.last_type_eaten = 0 #1,2,3, 0 if no dot eaten
        self.list_of_values_type_1 = []
        self.list_of_values_type_2 = []
        self.list_of_values_type_3 = []
        self.last_pacman_action = None
        self.steps = steps
        e1=0
        e2=0
        e3=0


    def update_expected_value_for_dot_types(self,accumulated_reward):

        diff = accumulated_reward - self.last_state_reward

        if self.last_type_eaten == 1:
            self.list_of_values_type_1 += [diff]
        elif self.last_type_eaten == 2:
            self.list_of_values_type_2 += [diff]
        elif self.last_type_eaten == 3:
            self.list_of_values_type_3 += [diff]

        self.last_state_reward = accumulated_reward
        pass

    def choose_next_action(self, state, accumulated_reward):
        # decrease step count
        self.steps -= 1

        """Choose next action for pacman controller given the current state.
        Action should be returned in the format described previous parts of the project.
        This method MUST terminate within the specified timeout.
        """
        # TODO: MUST ADD TIMER EFFECT

        # add value of type of dot to the correct list
        self.update_expected_value_for_dot_types(accumulated_reward)

        e1,e2,e3 = self.compute_expected_values()

        # Current PacmanState
        s_0 = PacmanState(state,self.last_pacman_action,e1,e2,e3)

        # If pacman dead - return reset.
        if "pacman" not in s_0.special_things:
            return "reset"
        if s_0.special_things["pacman"] == "dead":
            return "reset"


        # Generate possible future states (1 turn ahead).
        # {Pacman_action: {dictionary of ghost actions}}
        poss_state_dictionary = s_0.get_next_moves()
        # convert dictionary of ghost actions into 5-tuple

        possible_moves = convert_dictionary_to_tuple(poss_state_dictionary)

        all_states = set() # List of all states
        # Now we iterate through all possible moves.
        print(poss_state_dictionary)
        for move in possible_moves:
            print(move)
            # iterate through all possible permutations
            temp = c.deepcopy(s_0)

            temp.move_pacman_ghosts(move)
            all_states.add(temp)

        # Find max in all_states
        state = self.find_max_h(list(all_states))

        # update action chosen of winning state.
        self.last_pacman_action = state.last_pacman_action
        self.last_type_eaten = state.last_type_eaten

        return self.last_pacman_action

    def find_max_h(self,all_states):
        max_value = 0
        max_state = all_states[0]
        for s in all_states:
            if s.h_val >= max_value:
                max_state=s

        return max_state

    def create_list_of_all_possible_moves(ghost_move_dict):

        # returns list of all possible moves. l1= pacman.

        return [["R","R","R","R","R"],["R","R","R","R","L"],]

    def compute_expected_values(self):
        # Return expected values for dot 1, dot 2, dot 3

        # v1.0 = return mean inefficiently.
        if len(self.list_of_values_type_1)>0:
            e1 = sum(self.list_of_values_type_1)/len(self.list_of_values_type_1)
        else:
            e1 = 0
        if len(self.list_of_values_type_2)>0:
            e2 = sum(self.list_of_values_type_2)/len(self.list_of_values_type_2)
        else:
            e2 = 0

        if len(self.list_of_values_type_3)>0:
            e3 = sum(self.list_of_values_type_3)/len(self.list_of_values_type_3)
        else:
            e3 = 0

        # TODO: v2.0 guess distribution and return expected value.

        return e1,e2,e3




