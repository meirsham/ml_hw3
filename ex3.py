import time
import random
import checker


ids = ["000000000", "111111111"]


class PacmanState:

    def __init__(self,board_state,last_pacman_action=None,e1=0,e2=0,e3=0):
        # board
        self.state, self.special_things = checker.problem_to_state(board_state)
        self.last_pacman_action=last_pacman_action
        self.last_type_eaten = 0

        self.score = 0
        self.h_val = 0

        # Excpected Values for each type of dot
        self.e1=e1
        self.e2=e2
        self.e3=e3


    def move_pacman_ghosts(self,pacman_action,green_ghost=None,red_ghost=None,yellow_ghost=None,blue_ghost=None):
        # last type eaten is equal to 0 if no dot was eaten, 1 if 11, 2 if 12, 3 if 13
        # this is the previous value of Pacman's new position.
        self.last_type_eaten = 0

        # This function takes in the moves of Pacman + ghosts
        # and, updates the board.
        #
        #
        # Updates
        self.last_pacman_action = pacman_action


        # It doesn't return anything
        pass

    def compute_h(self):

        # This function computes the heuristic value of the current state.
        # The function doesn't return anything.
        # for example, h_val = score - (# of ghosts remaining)^2

        self.h_val = 0 #new value
        pass


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



    def choose_next_action(self, state, accumulated_reward):
        # decrease step count
        self.steps -= 1

        """Choose next action for pacman controller given the current state.
        Action should be returned in the format described previous parts of the project.
        This method MUST terminate within the specified timeout.
        """

        # add value of type of dot to the correct list
        diff = accumulated_reward - self.last_state_reward

        if self.last_type_eaten == 1:
            self.list_of_values_type_1 += [diff]
        elif self.last_type_eaten == 2:
            self.list_of_values_type_2 += [diff]
        elif self.last_type_eaten == 3:
            self.list_of_values_type_3 += [diff]

        self.last_state_reward = accumulated_reward

        e1,e2,e3 = self.compute_expected_values()

        # Current PacmanState
        s_0 = PacmanState(state,last_pacman_action = self.last_action,e1,e2,e3)

        for action in Pacman_moves:

            if action == U:
                pacman_location = s_0.special_things["pacman"] + (1,0) # blah
            elif action == D:
                pacman_location = s_0.special_things["pacman"] + (1, 0)  # blah
            elif action == R:
                pacman_location = s_0.special_things["pacman"] + (1,0) # blah
            elif action == L:
                pacman_location = s_0.special_things["pacman"] + (1,0) # blah
            else:
                print("ERROR")

            # For ghost in colors:
            ghost_move_dict = {}
            if ghost in s_0.special_things:
                ghost_location = s_0.special_things["ghost"]
                possible_moves = get_moves_based_on_md(ghost_location,pacman_location)
                ghost_move_dict[ghost]=possible_moves

        all_states = [] # List of all states
        # Now we iterate through all possible moves.
        #
        all_possible_moves = self.create_list_of_all_possible_moves(ghost_move_dict)
        for move in all_possible_moves:
            temp = deepcopy(s_0)
            temp.next_state(move)
            all_states+=temp

        # Find max in all_states
        state = self.find_max_h(all_states)

        # update action chosen of winning state.
        self.last_pacman_action = state.last_pacman_action

        # Set last type equal to eaten based on Pacman's chosen action.
        # We need this to know how to update the e1,e2,e3 lists next step.

        self.last_type_eaten = state.last_type_eaten


        return state.last_pacman_action

    def find_max_h(all_states):
        # returns max state

        return state

    def create_list_of_all_possible_moves(ghost_move_dict):

        # returns list of all possible moves. l1= pacman.

        return [["R","R","R","R","R"],["R","R","R","R","L"],]

    def get_moves_based_on_md(ghost_location, pacman_location):

        # return possible moves of ghost based on md from pacman


        return ["R","L"]


    def compute_expected_values(self):
        # Return expected values for dot 1, dot 2, dot 3
        e1 = sum(self.list_of_values_type_1)/len(self.list_of_values_type_1)

        e2 = sum(self.list_of_values_type_1)/len(self.list_of_values_type_2)

        e3 = sum(self.list_of_values_type_1)/len(self.list_of_values_type_3)
        return e1,e2,e3


        # compute all the possible states
        pacman_current_state = PacmanState(state)

        # compute the reward heuristic for each state.

            # This will be dynamic with the expected value based on the values in the
            # lists above {list_type_1} , {list_type_2} , {list_type_3}
            # at v1.0, we assume uniform distribute for each type and
            #    set expected value for each type to average value.
            #
            # at v2.0, we can guess the distribution for each type and then compute expected value.
            #
            #

        # choose the action that leads to the state with the highest reward heuristic

