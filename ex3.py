import time
import random
import checker
import copy as c
import scipy.stats as st
import warnings

import numpy as np
#import pandas as pd
import math as m
ids = ["000000000", "111111111"]
COLOR_ORDERED = ["blue","green","yellow","red"]

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
    if action == "UR":
        next_tile = (current_tile_x-1, current_tile_y - 1)
    if action == "DR":
        next_tile = (current_tile_x+1, current_tile_y - 1)
    if action == "UL":
        next_tile = (current_tile_x-1, current_tile_y + 1)
    if action == "DL":
        next_tile = (current_tile_x+1, current_tile_y + 1)

    if action == 0:
        next_tile = (current_tile_x, current_tile_y)
    return next_tile

def cal_diff(loc1,loc2):
    return [loc1[0]-loc2[0],loc1[1]-loc2[1]]

def get_eligible_moves(color,tile):
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
    if color is "blue" and len(moves) > 1:
        #"merge move bec blue can go diagnol.
        if "D" in moves and "L" in moves:
            moves = ["DL"]
        elif  "D" in moves and "R" in moves:
            moves = ["DR"]
        elif "U" in moves and "L" in moves:
            moves = ["UL"]
        elif "U" in moves and "R" in moves:
            moves = ["UR"]
    return moves

def get_list_values(color,dict_colors):
    if color in dict_colors:
        return dict_colors[color]
    else:
        return [0]


def convert_dictionary_to_tuple(poss_state_dictionary):
    # Done: Function receives a dictionary of dictionaries
    # Returns flat 5-tuple of states representing pacman, ghost actions
    # e.g.
    # poss_state_dictionary = {'U': {'green': ['U', 'L'], 'blue': ['D'],
    # 'yellow': ['L']}, 'R': {'green': ['L'], 'blue': ['D', 'R'], 'yellow': ['D']}, 'L': {'green': ['L'], 'blue': ['D', 'L'], 'yellow': ['D', 'L']}}
    #
    # returns: list of list of possible moves ordered by pacman, blue,yellow,green,red
    # 0 if doesn't exist
    # [["U","D","L","L",0],["U","D","L","L",0],["R"....etc.]]
    # Note for "Blue" - "DL" means down-left diagnol, "DR" means down-right diag, "UL" means up-left diag and "UR" means up-right diagnol.
    #
    # The code is naively written b/c of time constraints , and can definitely be improved by using recursion or simple functions.
    # pacman moves
    final_list = []
    for pac_action,color_dic in poss_state_dictionary.items():
        #pac_action # pacman's action
        #color_dic # key:value
        blue_list = get_list_values("blue",color_dic)
        green_list = get_list_values("green",color_dic)
        yellow_list = get_list_values("yellow",color_dic)
        red_list = get_list_values("red",color_dic)

        full_list = [[pac_action,blue_list[0],green_list[0],yellow_list[0],red_list[0]]]

        for i,a_list in enumerate([blue_list,green_list,yellow_list,red_list]):
            if len(a_list) == 2:
                temp_list= c.deepcopy(full_list)
                for inside_temp_list in temp_list:
                    inside_temp_list[i+1] = a_list[1]
                full_list += temp_list


        final_list +=full_list


    return final_list

class DistributionType:

    def __init__(self,dot_count):

        self.distribution = st.norm
        self.best_sse = np.inf
        self.best_params = (0.0, 1.0)
        self.dist_iter = 0
        self.list_of_values = []
        self.exp_value = 1
        self.flag = False # Did we add a new dot or not.
        self.dot_count = dot_count # number of dots of this type on board
        self.N = max(10,dot_count*0.10)
    def add_dot(self,data):
        self.list_of_values += [data]
        return

    def set_flag(self,bool):
        self.flag = bool
        return
    def compute_mean(self):
        self.exp_value = sum(self.list_of_values)/len(self.list_of_values)
        self.flag = False
        return
class PacmanState:

    def __init__(self,board_state,last_pacman_action="reset",e1=0,e2=0,e3=0):
        # board
        self.state, self.special_things = checker.problem_to_state(board_state)
        self.size_of_board = len(self.state)
        self.last_pacman_action=last_pacman_action
        self.last_type_eaten = 0
        self.score = 0
        self.h_val = 0
        self.future_h_val = 0
        self.numb_of_ghosts = self.get_num_ghosts()
        # Expected Values for each type of dot
        self.e1=e1
        self.e2=e2
        self.e3=e3
        self.numb_of_dots_within_three = self.get_dot_neighbors()
        self.numb_of_dots_div_board_size = self.numb_of_dots_within_three / self.size_of_board
        # initialize location of all the dots
        self.list_of_dots = {}
        for number_of_row, row in enumerate(board_state):
            for number_of_column, cell in enumerate(row):
                if cell%10 == 1 or cell%10 == 2 or cell%10 == 3:
                    self.list_of_dots[(number_of_row, number_of_column)] = cell

    def get_size_of_board(self):
        len(self.state)

    def get_num_ghosts(self):
        i = 0
        for k,j in checker.COLOR_CODES.items():
            if k in self.special_things:
                i+=1
        return i

    def dist_to_pacman(self,item, pacman):
        a = item[0]
        b = item[1]
        x = pacman[0]
        y = pacman[1]
        md = m.fabs(a-x)+m.fabs(b-y)
        return md

    def get_dot_neighbors(self):
        exp_value = 0
        explored_squares = 0
        if "pacman" not in self.special_things or self.special_things["pacman"] is "dead":
            return -10
        new_x,new_y= self.special_things["pacman"]

        pacman = (new_x, new_y)

        # Check left of pacman for sensors
        queue = [[new_x,new_y]]
        j=0
        while j < 4 and len(queue) > 0:
            new_x,new_y=queue.pop()
            explored_squares+=1
            # left
            ul = [new_x-1,new_y-1]
            l = [new_x,new_y-1]
            dl = [new_x+1,new_y-1]
            list_of_points = [ul,l,dl]

            for item in list_of_points:
                flag = True
                x=item[0]
                y=item[1]
                val = self.state[item[0],item[1]]
                if val == 99:
                    flag = False

                elif self.state[item[0],item[1]] is 11:
                    exp_value += self.e1
                elif self.state[item[0], item[1]] is 12:
                    exp_value += self.e2
                elif self.state[item[0], item[1]] is 13:
                    exp_value += self.e3
                elif self.state[item[0], item[1]] in checker.BLOCKING_CODES:
                    #exp_value -= 2.5 /self.dist_to_pacman(item,[new_x,new_y])
                    printable = 2.5 /self.dist_to_pacman(item,[new_x,new_y])
                    #print(printable)
                    exp_value-=printable
            if flag:
                queue.append(l)

            # right
            ur = [new_x - 1, new_y + 1]
            r = [new_x, new_y + 1]
            dr = [new_x + 1, new_y + 1]
            list_of_points = [dr, r, ur]

            for item in list_of_points:
                flag = True
                val = self.state[item[0],item[1]]
                if val == 99:
                    flag = False
                elif self.state[item[0], item[1]] is 11:
                    exp_value += self.e1
                elif self.state[item[0], item[1]] is 12:
                    exp_value += self.e2
                elif self.state[item[0], item[1]] is 13:
                    exp_value += self.e3
                elif self.state[item[0], item[1]] in checker.BLOCKING_CODES:
                    exp_value -= 2.5 / self.dist_to_pacman(item, pacman)

            if flag:
                queue.append(r)


            u = [new_x-1,new_y]
            if self.state[u[0], u[1]] is 11:
                exp_value += self.e1
            elif self.state[u[0], u[1]] is 12:
                exp_value += self.e2
            elif self.state[u[0], u[1]] is 13:
                exp_value += self.e3
            elif self.state[item[0], item[1]] in checker.BLOCKING_CODES:
                exp_value -= 2.5 / self.dist_to_pacman(item,pacman)

            if self.state[u[0],u[1]] is not 99:
                queue.append(u)

            d = [new_x,new_y+1]
            if self.state[d[0], d[1]] is 11:
                exp_value += self.e1
            elif self.state[d[0], d[1]] is 12:
                exp_value += self.e2
            elif self.state[d[0], d[1]] is 13:
                exp_value += self.e3
            elif self.state[item[0], item[1]] in checker.COLOR_CODES:
                exp_value -= 3 * 1 / self.dist_to_pacman(item, pacman)

            if self.state[d[0],d[1]] is not 99:
                queue.append(d)

            j+=1

        return exp_value/max(9,explored_squares)

    def move_pacman_to_walkable_tile(self, current_tile, next_tile):
        if self.state[next_tile] == 11:
            #
            self.score += self.e1


        if self.state[next_tile] == 12:
            self.score += self.e2
        if self.state[next_tile] == 13:
            self.score += self.e3

        self.state[next_tile] = 66
        self.state[current_tile] = 10
        self.special_things["pacman"] = next_tile

    def move_pacman_ghosts(self,move):
        pacman_action = move[0]
        ghost_moves = move[1:]
        self.move_pacman(pacman_action)
        self.move_ghosts(ghost_moves)

    def move_ghosts(self,ghost_moves):

        for j,color in enumerate(["blue","green","yellow","red"]):
            action = ghost_moves[j]
            if action is 0 or color not in self.special_things:
                pass
            else:
                ghost_place_x, ghost_place_y = self.special_things[color]
                current_tile = self.special_things[color]
                previous_tile_pill_number = self.state[current_tile] % 10
                previous_tile_contained_pill = 1 <= previous_tile_pill_number <= 3
                ghost_code = checker.COLOR_CODES[color]


                next_tile = get_next_tile(current_tile,ghost_moves[j])

                # movement result

                # next tile is a regular tile (with or without a pill)
                if 10 <= self.state[next_tile] <= 13:
                    self.state[next_tile] = ghost_code + (self.state[next_tile] % 10)
                    self.special_things[color] = next_tile

                # poison
                elif self.state[next_tile] == 77 or 70 <= self.state[next_tile] <= 73:
                    self.numb_of_ghosts-=1
                    if self.state[next_tile] == 77:
                        self.state[next_tile] = 10
                    else:
                        self.state[next_tile] = 10 + previous_tile_pill_number
                    del self.special_things[color]

                # ghost got the pacman
                elif self.state[next_tile] == 66 or self.state[next_tile] == 88:
                    self.special_things["pacman"] = "dead"
                    self.state[next_tile] = 88
                    self.score = -5

                if current_tile != next_tile:
                    if previous_tile_contained_pill:
                        self.state[current_tile] = 10 + previous_tile_pill_number
                    else:
                        self.state[current_tile] = 10

    def move_pacman(self,action):
        self.last_pacman_action = action

        # actions_list is ordered by pacman, blue, yellow, green, red
        # 0 if no action taken / ghost doesn't exist
        # ["U","D","L","L",0]
        # means that pacman moves U, blue ghost moves D, green moves L, yellow moves L, red doesn't exist.

        next_tile = None
        current_tile_x, current_tile_y = self.special_things["pacman"]

        if action == "U":
            next_tile = (current_tile_x - 1, current_tile_y)
        if action == "R":
            next_tile = (current_tile_x, current_tile_y + 1)
        if action == "D":
            next_tile = (current_tile_x + 1, current_tile_y)
        if action == "L":
            next_tile = (current_tile_x, current_tile_y - 1)

        assert next_tile is not None
        self.last_type_eaten = self.state[next_tile]%10

        # wall
        if self.state[next_tile] == 99:
            return

        # walkable tile
        if self.state[next_tile] in checker.WALKABLE_TILES:
            self.move_pacman_to_walkable_tile((current_tile_x, current_tile_y), next_tile)
            return

        # ghosts and poison
        if self.state[next_tile] in checker.LOSS_INDEXES:
            self.state[next_tile] = 88
            self.score = -5
            self.state[(current_tile_x, current_tile_y)] = 10
            self.special_things["pacman"] = "dead"

        return

    def compute_h(self):
        # TODO
        # This function computes the heuristic value of the current state.
        # The function doesn't return anything.
        # for example, h_val = score - (# of ghosts remaining)^2
        # md to nearest dot
        estimated_reward = 0
        next_tile = None
        if "pacman" in self.special_things and self.special_things["pacman"] != "dead":
            current_tile_x, current_tile_y = self.special_things["pacman"]
            for num_of_steps in range(0,10):
                #current_tile_x, current_tile_y = self.special_things["pacman"]
                [estimated_reward, current_tile_x, current_tile_y] = self.estimate_best_path_rewrad(estimated_reward, current_tile_x, current_tile_y)

        self.exp_value_nearby = self.get_dot_neighbors()

        self.h_val = 10*self.score - 10*self.numb_of_ghosts**2 + self.exp_value_nearby + 10*estimated_reward   #new value


        pass

    def estimate_best_path_rewrad(self, estimated_reward, current_tile_x, current_tile_y):
        min_md = 2**32-1
        possible_reward = 0
        # measure MD to all pills
        next_tile = None
        for action in {"U", "D", "L", "R"}:
            #current_tile_x, current_tile_y = self.special_things["pacman"]
            if action == "U":
                next_tile = (current_tile_x - 1, current_tile_y)
            if action == "R":
                next_tile = (current_tile_x, current_tile_y + 1)
            if action == "D":
                next_tile = (current_tile_x + 1, current_tile_y)
            if action == "L":
                next_tile = (current_tile_x, current_tile_y - 1)
            # wall
            if next_tile  not in self.state.keys():
                continue
            assert next_tile is not None
            if self.state[next_tile] not in checker.WALKABLE_TILES:
                continue
            for cell_dot in self.list_of_dots:
                # make sure the dot still exists
                if self.state[cell_dot] == 11  and self.e1 > 0:
                    cur_md = abs(next_tile[0]-cell_dot[0]) + abs(next_tile[1]-cell_dot[1])
                    if cur_md < min_md :
                        min_md = cur_md
                        possible_reward = self.whats_next(next_tile, self.e1)
                elif self.state[cell_dot] == 12 and self.e2 > 0:
                    cur_md = abs(next_tile[0]-cell_dot[0]) + abs(next_tile[1]-cell_dot[1])
                    if cur_md < min_md :
                        min_md = cur_md
                        possible_reward = self.whats_next(next_tile, self.e2)
                elif self.state[cell_dot] == 13 and self.e3 > 0:
                    cur_md = abs(next_tile[0]-cell_dot[0]) + abs(next_tile[1]-cell_dot[1])
                    if cur_md < min_md :
                        min_md = cur_md
                        possible_reward = self.whats_next(next_tile, self.e3)

        estimated_reward+=possible_reward
        return [estimated_reward, next_tile[0], next_tile[1]]

    def whats_next(self, next_tile, dot_val):
        if self.state[next_tile] == 10:
            return 0
        else:
            return dot_val

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
        # TO DO: Returns dictionary of all ghosts and their possible moves
        ghost_dict = {}

        #pacman_location = self.special_things["pacman"]
        for color in checker.COLOR_CODES:
            if color in self.special_things:
                # find possible moves for the special things
                ghost_location = self.special_things[color]
                location_diff = cal_diff(next_pacman_location,ghost_location)
                moves = get_eligible_moves(color,location_diff)
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

    def set_real_h_value(self,round_2_h_val,gamma=0.9):
        self.future_h_val = self.h_val+gamma*round_2_h_val
        return

class PacmanController:
    """This class is a controller for a pacman agent."""
    def __init__(self, state, steps):
        """Initialize controller for given the initial setting.
        This method MUST terminate within the specified timeout."""
        # print('COMPLETE init ')

        # Keep track of the old value
        self.state = state  # original state.
        self.last_state_reward = 0
        self.last_type_eaten = 0 #1,2,3, 0 if no dot eaten
        self.last_pacman_action = None
        self.steps = steps
        self.rounds = 0
        self.last_time_round = 0

        self.distribution = st.norm
        # Need to create different distributions for each type of ball. {1: st.norm, 2: st.norm, 3: st.norm}
        self.state = state #original state.
        self.dots_count = self.eval_dots()
        #print(self.dots_count)
        #new stuff
        self.type_1 = DistributionType(self.dots_count[1])
        self.type_2 = DistributionType(self.dots_count[2])
        self.type_3 = DistributionType(self.dots_count[3])

    def eval_dots(self):
        values = list(self.state)
        d = {1:0,2:0,3:0}
        for i in values:
            for j in i:
                value = j%10

                if value < 4:

                    d[value]+=1
        return d

    def get_type_lists(self):
        print(self.type_1.list_of_values)
        print(self.type_2.list_of_values)
        print(self.type_3.list_of_values)
        return
    def update_expected_value_for_dot_types(self,accumulated_reward):
        diff = accumulated_reward - self.last_state_reward
        #print("Difference %s, Last Action %s " % (diff,self.last_pacman_action))

        if self.last_pacman_action is "reset":
            if diff is -5:
                return
            else:
                # adds 5 to get the real value of the dot.
                diff+=5

        if self.last_type_eaten == 1:
            self.type_1.add_dot(diff)
            self.type_1.set_flag(True)


        elif self.last_type_eaten == 2:
            self.type_2.add_dot(diff)
            self.type_2.set_flag(True)

        elif self.last_type_eaten == 3:
            self.type_3.add_dot(diff)
            self.type_3.set_flag(True)

        self.last_state_reward = accumulated_reward

        pass

    def print_the_type_summary(self):
        print(self.type_1.distribution, self.type_1.exp_value, self.type_1.list_of_values)
        print(self.type_2.distribution, self.type_2.exp_value, self.type_2.list_of_values)
        print(self.type_3.distribution, self.type_3.exp_value, self.type_3.list_of_values)
        print()
        print(self.dots_count)
        return

    def choose_next_action(self, state, accumulated_reward):

        self.steps -= 1
        if self.steps<1:
            self.print_the_type_summary()

        if self.rounds is 0:
            self.last_time_round=time.time()
            diff = 0
        else:
            temp_time = time.time()
            diff = temp_time-self.last_time_round
            self.last_time_round = temp_time
        #print("Round %s, Time: %s, length: %s " % (self.rounds, time.time(), diff))
        """Choose next action for pacman controller given the current state.
            Action should be returned in the format described previous parts of the project.
            This method MUST terminate within the specified timeout (5 seconds)
        """

        # decrease step count
        self.rounds +=1
        # if self.rounds % 5 ==0:
        #     print("e1 %s %s, e2 %s %s, e3 %s %s" % (self.type_1.exp_value, self.type_1.flag, self.type_2.exp_value, self.type_2.flag,
        #                                             self.type_3.exp_value, self.type_3.flag))
        # # Current PacmanState # we should've use inheritance ;-)
        s_0 = PacmanState(state, self.last_pacman_action, self.type_1.exp_value, self.type_2.exp_value,
                          self.type_3.exp_value)

        # If pacman dead - return reset.
        if "pacman" not in s_0.special_things or s_0.special_things["pacman"] == "dead":
            self.last_pacman_action = "reset"
            return self.last_pacman_action

        # add value of type of dot to the correct list
        if self.state is not state:
            self.update_expected_value_for_dot_types(accumulated_reward)

        # Rules
        #   if >= 8 dots of a type, so fit for a distribution each time, if we added a new dot.
        #   if < 8 dots of a type, so compute the mean, if we added a new dot.

        for type in [self.type_1,self.type_2,self.type_3]:
            # N = max(10, 10 % of the type of dots on the board)
            if len(type.list_of_values) >= type.N and type.flag:
                self.compute_expected_values_plus_new(type)
                type.set_flag(False)

        if self.rounds <= 100:

            for type in [self.type_1, self.type_2, self.type_3]:

                if len(type.list_of_values) > 4 and len(type.list_of_values) < 8 and type.flag:

                    self.compute_expected_values_mean(type)

                #print("updated all")
            #self.compute_expected_values_mean()

                # Current PacmanState # we should've use inheritance ;-)
        s_0 = PacmanState(state,self.last_pacman_action,self.type_1.exp_value,self.type_1.exp_value,self.type_1.exp_value)

        all_next_round_possible_states = self.get_next_round_states(s_0)

        # Find max in all_states

        # Second round
        for each_state in all_next_round_possible_states:
            if (time.time() - self.last_time_round ) > 4.9:
                # break for-loop and choose a random action to return
                break
            round_2_states = []
            if each_state.special_things["pacman"] is not "dead":
                round_2_states = list(self.get_next_round_states(each_state))
            temp_state = self.find_max_h(round_2_states)
            if temp_state is not False:
                each_state.set_real_h_value(temp_state.h_val)

        # Find max "future_h_value"

        state = self.find_max_future_h_value(list(all_next_round_possible_states))
        if state:
            # Update action chosen of winning state.
            self.last_pacman_action = state.last_pacman_action
        else:
            return "reset"
        self.last_type_eaten = state.last_type_eaten
        return self.last_pacman_action

    def get_next_round_states(self,s_0):
        # s_0 = initial state.
        # Generate possible future states (1 turn ahead).

        # {Pacman_action: {dictionary of ghost actions}}
        poss_state_dictionary = s_0.get_next_moves()

        # convert dictionary of ghost actions into 5-tuple
        possible_moves = convert_dictionary_to_tuple(poss_state_dictionary)

        all_states = set()  # List of all states
        # Now we iterate through all possible moves.

        for move in possible_moves:
            # iterate through all possible permutations
            temp = c.deepcopy(s_0)

            temp.move_pacman_ghosts(move)
            temp.compute_h()
            all_states.add(temp)
        return all_states

    def find_max_h(self,all_states):
        if len(all_states) is 0:
            return False #Need to reset
        max_value = all_states[0].h_val
        max_state = all_states[0]
        for s in all_states:
            if s.h_val >= max_value:
                max_value = s.h_val
                max_state=s

        return max_state

    def find_max_future_h_value(self,all_states):
        if len(all_states) is 0:
            return False  # Need to reset
        max_value = all_states[0].future_h_val
        max_state = all_states[0]
        for s in all_states:
            if s.future_h_val >= max_value:
                max_value = s.future_h_val
                max_state = s

        return max_state

    def compute_expected_values_plus_new(self,type):
        data = type.list_of_values
        self.compute_expected_values_guess_distribution(data,type)
        return

    def compute_expected_values_guess_distribution(self,data,type):
        # TO DO find best distribution fit for each dot type based on 11 observations
        # Set e1 = expected value of each distribution. Some code / strategy copied from here:
        # https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python?lq=1
        # Distributions to check
        # NOT CURRENTLY USING THIS BECAUSE THE TIME CONSTRAINT OF 5 SEC PER TURN.

        DISTRIBUTIONS_1 = [st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2]
        DISTRIBUTIONS_2 = [st.cosine,st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife,
            st.fisk,st.foldcauchy, st.foldnorm]
        DISTRIBUTIONS_3 = [st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm,
            st.genexpon,st.genextreme, st.gausshyper, st.gamma, st.gengamma]
        DISTRIBUTIONS_4 = [st.genhalflogistic, st.gilbrat, st.gompertz,
            st.gumbel_r,st.gumbel_l, st.halfcauchy, st.halflogistic]
        DISTRIBUTIONS_5 = [st.genhalflogistic, st.gilbrat, st.gompertz,
            st.gumbel_r,st.gumbel_l, st.halfcauchy, st.halflogistic]
        DISTRIBUTIONS_6 = [st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma,
            st.invgauss,st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone]
        DISTRIBUTIONS_7 = [st.kstwobign, st.laplace, st.levy, st.levy_l,
            st.levy_stable,st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax]
        DISTRIBUTIONS_8 = [st.kstwobign, st.laplace, st.levy, st.levy_l,
            st.levy_stable,st.logistic, st.loggamma]
        DISTRIBUTIONS_9 = [st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2,
            st.ncf,st.nct, st.norm]
        #DISTRIBUTIONS_10 = [st.pareto, st.pearson3,st.reciprocal,st.rayleigh, st.rice, st.recipinvgauss]
        #DISTRIBUTIONS_11 = [st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm,st.tukeylambda,st.uniform]
        #DISTRIBUTIONS_12 = [st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy]
        # ALL_DISTRIBUTIONS = [
        #     DISTRIBUTIONS_1,
        #     DISTRIBUTIONS_2,
        #     DISTRIBUTIONS_3,
        #     DISTRIBUTIONS_4,
        #     DISTRIBUTIONS_5,
        #     DISTRIBUTIONS_6,
        #     DISTRIBUTIONS_7,
        #     DISTRIBUTIONS_8,
        #     DISTRIBUTIONS_9,
        #     DISTRIBUTIONS_10,
        #     DISTRIBUTIONS_11,
        #     DISTRIBUTIONS_12
        # ]
        # # for testing purposes
        DISTRIBUTIONS = [st.uniform,st.gamma,\
                         st.expon, st.exponnorm,\
                         st.lognorm,st.powerlaw,\
                         st.powerlognorm, st.powernorm,\
                         st.rdist, st.norm,\
                         st.pareto,\
                         st.laplace]

        # Estimate distribution parameters from data
        type.dist_iter+=1

        #DIST_LIST = ALL_DISTRIBUTIONS[type.dist_iter%12]
        DIST_LIST = DISTRIBUTIONS
        for distribution in DIST_LIST:
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    warnings.simplefilter("ignore")

                    params = distribution.fit(data)
                    #print("distribution %s" % distribution)
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Get histogram of original data
                    y, x = np.histogram(data, bins=10, density=True)
                    x = (x + np.roll(x, -1))[:-1] / 2.0


                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = m.fabs(np.sum(np.power(y - pdf, 2.0)))

                    # identify if this distribution is better
                    #sse = 0
                    if type.best_sse > sse:
                        type.distribution= distribution
                        type.best_params = params
                        type.best_sse = sse
                        type.exp_value = distribution.expect(None,arg,loc,scale)

            except Exception:
                pass

        #print("best_distribution %s" % self.distribution)
        return

    def compute_expected_values_mean(self,type):
        if len(type.list_of_values)>0:
            type.compute_mean()


