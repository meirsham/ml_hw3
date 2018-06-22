import time
import random
import checker
import copy as c

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


class PacmanState:

    def __init__(self,board_state,last_pacman_action=None,e1=0,e2=0,e3=0):
        # board
        self.state, self.special_things = checker.problem_to_state(board_state)
        self.size_of_board = len(self.state)
        self.last_pacman_action=last_pacman_action
        self.last_type_eaten = 0
        self.score = 0
        self.h_val = 0
        self.future_h_val = 0
        self.numb_of_ghosts = self.get_num_ghosts()
        self.numb_of_dots_within_three = self.get_dot_neighbors()
        self.numb_of_dots_div_board_size = self.numb_of_dots_within_three/self.size_of_board

        # Expected Values for each type of dot
        self.e1=e1
        self.e2=e2
        self.e3=e3

    def get_size_of_board(self):
        len(self.state)

    def get_num_ghosts(self):
        i = 0
        for k,j in checker.COLOR_CODES.items():
            if k in self.special_things:
                i+=1
        return i

    def get_dot_neighbors(self):
        count = 0
        if "pacman" not in self.special_things:
            return -10

        new_x,new_y= self.special_things["pacman"]
        # Check left of pacman for sensors
        queue = [[new_x,new_y]]
        j=0
        while j < 3:
            new_x,new_y=queue.pop()
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

                elif self.state[item[0],item[1]] in (11, 12, 13):
                    count += 1
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

                elif self.state[item[0],item[1]] in (11, 12, 13):
                    count += 1
            if flag:
                queue.append(r)


            u = [new_x-1,new_y]
            if self.state[u[0],u[1]] in (11,12,13):
                count+=1
            if self.state[u[0],u[1]] is not 99:
                queue.append(u)

            d = [new_x,new_y+1]
            if self.state[d[0],d[1]] in (11, 12, 13):
                count += 1
            if self.state[d[0],d[1]] is not 99:
                queue.append(d)

            j+=1

        return count

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
        self.compute_h()

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

        self.h_val = 10*self.score-10*self.numb_of_ghosts + 2*self.numb_of_dots_div_board_size  #new value

        pass

    def get_possible_moves(self,special_thing):
        # pacman is alive and cannot move into a wall or a ghost or a poison!
        special_thing_moves = []
        current_tile = self.special_things[special_thing]
        print(current_tile)
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
        self.last_state_reward = 0
        self.last_type_eaten = 0 #1,2,3, 0 if no dot eaten
        self.list_of_values_type_1 = []
        self.list_of_values_type_2 = []
        self.list_of_values_type_3 = []
        self.last_pacman_action = None
        self.steps = steps
        self.rounds = 0
        self.e1=1
        self.e2=1
        self.e3=1


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
        self.rounds +=1
        self.steps -= 1

        """Choose next action for pacman controller given the current state.
        Action should be returned in the format described previous parts of the project.
        This method MUST terminate within the specified timeout.
        """
        # TODO: MUST ADD TIMER EFFECT

        # add value of type of dot to the correct list
        if self.rounds <= 10:
            self.update_expected_value_for_dot_types(accumulated_reward)
            self.compute_expected_values_mean()
        # TODO Imporve the way we compute expected values of dot types
        # if self.rounds == 11:
        #   self.compute_expected_values_plus()

        # Current PacmanState # we should use inheritance ;-)
        s_0 = PacmanState(state,self.last_pacman_action,self.e1,self.e2,self.e3)

        # If pacman dead - return reset.
        if "pacman" not in s_0.special_things:
            return "reset"
        if s_0.special_things["pacman"] == "dead":
            return "reset"

        all_next_round_possible_states = self.get_next_round_states(s_0)

        # Find max in all_states

        # second round
        for each_state in all_next_round_possible_states:
            round_2_states = []
            if each_state.special_things["pacman"] is not "dead":
                round_2_states = list(self.get_next_round_states(each_state))
            temp_state = self.find_max_h(round_2_states)
            if temp_state is not False:
                each_state.set_real_h_value(temp_state.h_val)

        # Find max "future_h_value"

        state = self.find_max_future_h_value(list(all_next_round_possible_states))
        # update action chosen of winning state.
        self.last_pacman_action = state.last_pacman_action
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
            all_states.add(temp)
        return all_states

    def find_max_h(self,all_states):
        if len(all_states) is 0:
            return False #Need to reset
        max_value = all_states[0].h_val
        max_state = all_states[0]
        for s in all_states:
            if s.h_val >= max_value:
                max_state=s

        return max_state

    def find_max_future_h_value(self,all_states):
        if len(all_states) is 0:
            return False  # Need to reset
        max_value = all_states[0].future_h_val
        max_state = all_states[0]
        for s in all_states:
            if s.h_val >= max_value:
                max_state = s

        return max_state

    def compute_expected_values_plus(self):
        # TODO find best distribution fit for each dot type based on 11 observations
        # Set e1 = expected value of each distribution.
        # HINT: https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python?lq=1
        return

    def compute_expected_values_mean(self):
        # Return expected values for dot 1, dot 2, dot 3

        # v1.0 = return mean inefficiently.
        if len(self.list_of_values_type_1)>0:
            self.e1 = sum(self.list_of_values_type_1)/len(self.list_of_values_type_1)
        else:
            self.e1 = 0
        if len(self.list_of_values_type_2)>0:
            self.e2 = sum(self.list_of_values_type_2)/len(self.list_of_values_type_2)
        else:
            self.e2 = 0

        if len(self.list_of_values_type_3)>0:
            self.e3 = sum(self.list_of_values_type_3)/len(self.list_of_values_type_3)
        else:
            self.e3 = 0

        # TODO: v2.0 guess distribution and return expected value.

        return




