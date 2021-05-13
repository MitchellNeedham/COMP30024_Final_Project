# COMP300024 Final Project
# Authors:
#  - Mitchell Needham  : 823604
#  - Ed Hinrichsen     : 993558

import random
import gametheory
import copy
import numpy as np
import json
# import math


from pprint import pprint

# ------------ GAME CONSTANTS ------------ #
TOKEN_TO_ATTACK = {"r": "s", "p": "r", "s": "p"}
TOKEN_TYPES = ["r", "p", "s"]
INIT_TOKEN_COUNT = 9
BOARD_SIZE = 4
UCB1_C = 2


class Player:
    # ------------ GAME STATE ------------ #
    player_remaining_tokens = INIT_TOKEN_COUNT
    opponent_remaining_tokens = INIT_TOKEN_COUNT
    player_type = ""
    opponent_type = ""
    board_state = None
    player_tokens = list()
    opponent_tokens = list()

    # ------------ MOVE DATA ------------ #
    player_available_moves = []
    opponent_available_moves = []

    def __init__(self, player):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "upper" (if the instance will
        play as Upper), or the string "lower" (if the instance will play
        as Lower).
        """
        # put your code here
        self.player_type = player
        self.opponent_type = 'lower' if player == 'upper' else 'upper'
        self.board_state = init_board(BOARD_SIZE)
        self.player_available_moves = get_available_throws(
            self.board_state, self.player_remaining_tokens, player)

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        # put your code here
        # return random.choice(self.available_moves)
        return get_payoff(self.player_available_moves, self.opponent_available_moves, self.board_state, self.player_type)

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        # put your code here

        self.player_available_moves = []
        self.opponent_available_moves = []

        if player_action[0] == "THROW":
            self.player_remaining_tokens -= 1

        if opponent_action[0] == "THROW":
            self.opponent_remaining_tokens -= 1

        self.board_state, self.player_tokens, self.opponent_tokens = \
            update_board(player_action,
                         opponent_action,
                         self.player_tokens,
                         self.opponent_tokens,
                         self.board_state,
                         self.player_type)

        self.player_available_moves = get_available_moves(self.board_state,
                                                          self.player_tokens,
                                                          self.opponent_tokens,
                                                          self.player_remaining_tokens,
                                                          self.player_type)

        self.opponent_available_moves = get_available_moves(self.board_state,
                                                            self.opponent_tokens,
                                                            self.player_tokens,
                                                            self.opponent_remaining_tokens,
                                                            self.opponent_type)


class Tile:
    def __init__(self, r, q):
        self.pos = (r, q)
        self.tokens = []
        self.neighbours = [valid_path(r, q - 1),
                           valid_path(r, q + 1),
                           valid_path(r + 1, q - 1),
                           valid_path(r + 1, q),
                           valid_path(r - 1, q),
                           valid_path(r - 1, q + 1)]


class Token:
    def __init__(self, token_type, pos):
        self.pos = pos
        self.type = token_type

    def update_position(self, pos):
        self.pos = pos


# ---------------------- HELPER FUNCTIONS ------------------------ #
def init_board(board_size):
    board_dict = dict()
    r_range = range(-board_size, board_size + 1)
    q_range = range(-board_size, board_size + 1)
    for r in r_range:
        for q in q_range:
            if abs(r + q) <= board_size:
                board_dict[(r, q)] = Tile(r, q)
    return board_dict


def update_board(player_action, opponent_action, player_tokens, opponent_tokens, board_state, player_type):
    if player_action[0] == "THROW":
        token_type = player_action[1] if player_type == "lower" else player_action[1].upper(
        )
        new_token = Token(token_type, player_action[2])

        # print(True, player_type, new_token.pos)
        player_tokens.append(new_token)
        # update game state
        board_state[player_action[2]].tokens.append(new_token)

    else:
        # get token object to move
        tile_tokens = board_state[player_action[1]].tokens
        token = list(filter(lambda t: t.type.isupper() ==
                            (player_type == "upper"), tile_tokens))[0]

        # change token position
        token.update_position(player_action[2])

        # update board
        board_state[player_action[1]].tokens.remove(token)
        board_state[player_action[2]].tokens.append(token)

    if opponent_action[0] == "THROW":
        token_type = opponent_action[1] if player_type != "lower" else opponent_action[1].upper(
        )
        new_token = Token(token_type, opponent_action[2])

        # print(False, player_type, new_token.pos)
        opponent_tokens.append(new_token)
        # update game state
        board_state[opponent_action[2]].tokens.append(new_token)

    else:
        # get token object to move
        tile_tokens = board_state[opponent_action[1]].tokens
        token = list(filter(lambda t: t.type.isupper() ==
                            (player_type != "upper"), tile_tokens))[0]

        # change token position
        token.update_position(opponent_action[2])

        # update board
        board_state[opponent_action[1]].tokens.remove(token)
        board_state[opponent_action[2]].tokens.append(token)
    board_state = handle_collision(board_state, player_action[2])
    board_state = handle_collision(board_state, opponent_action[2])

    return board_state, player_tokens, opponent_tokens


def handle_collision(board_state, pos):
    tokens = board_state[pos].tokens
    if len(tokens) < 2:
        return board_state

    to_destroy = []

    for attacker_token in tokens.copy():
        for attacked_token in tokens.copy():
            can_attack = (
                TOKEN_TO_ATTACK[attacker_token.type.lower()] == attacked_token.type.lower())
            if can_attack and attacked_token not in to_destroy:
                to_destroy.append(attacked_token)
                break

    for token in to_destroy:
        board_state[pos].tokens.remove(token)

    return board_state


def valid_path(r, q):
    if r > BOARD_SIZE or q > BOARD_SIZE or r < -BOARD_SIZE or q < -BOARD_SIZE or abs(r + q) > BOARD_SIZE:
        return None
    return r, q


def get_available_moves(board_state, player_tokens, opponent_tokens, remaining_tokens, player_type):
    available_moves = []
    collision_locations = map(lambda x: x.pos, opponent_tokens)
    # print(list(map(lambda x: x.pos, player_tokens)),
    #       player_type, hex(id(player_tokens)))

    for token in player_tokens:
        neighbour_tiles = filter(None, board_state[token.pos].neighbours)
        swing_locations = filter(None, get_swing_locations(board_state, token))

        safe_neighbours = list(
            set(neighbour_tiles).difference(collision_locations))
        safe_swing_locations = list(set(swing_locations).difference(
            neighbour_tiles).difference(collision_locations))

        available_moves += map(lambda x: ("SLIDE",
                                          token.pos, x), safe_neighbours)
        available_moves += map(lambda x: ("SWING",
                                          token.pos, x), safe_swing_locations)

    available_moves += get_available_throws(board_state,
                                            remaining_tokens, player_type)

    return available_moves


def get_available_throws(board_state, remaining_tokens, player_type):
    if remaining_tokens <= 0:
        return []

    available_throws = []

    if player_type == "upper":
        r_range = range(-BOARD_SIZE + remaining_tokens - 1, BOARD_SIZE + 1)
    else:
        r_range = range(-BOARD_SIZE, BOARD_SIZE - remaining_tokens + 2)

    for r in r_range:
        for q in range(-BOARD_SIZE, BOARD_SIZE):
            if not valid_path(r, q):
                continue
            for token_type in TOKEN_TYPES:
                available_throws.append(("THROW", token_type, (r, q)))
    return available_throws


def get_swing_locations(board_state, token):
    swing_locations = []
    neighbours = filter(None, board_state[token.pos].neighbours)
    for tile in neighbours:
        for neighbour_token in board_state[tile].tokens:
            if neighbour_token.type.isupper() and token.type.isupper():
                continue
            if neighbour_token.type.islower() and token.type.islower():
                continue
            for swing_tile in board_state[neighbour_token.pos].neighbours:
                if swing_tile in neighbours or swing_tile == token.pos:
                    continue
                swing_locations.append(swing_tile)
    return list(set(swing_locations))


# -------------------------------  board_state = copy.deepcopy(board_state)

def update_board_non_destructive(player_action, opponent_action, player_tokens, opponent_tokens, board_state, player_type):

    
    player_tokens = copy.deepcopy(player_tokens)
    opponent_tokens = copy.deepcopy(opponent_tokens)
    board_state = copy.deepcopy(board_state)
     

    if player_action[0] == "THROW":
        token_type = player_action[1] if player_type == "lower" else player_action[1].upper(
        )
        new_token = Token(token_type, player_action[2])

        # print(True, player_type, new_token.pos)
        player_tokens.append(new_token)
        # update game state
        board_state[player_action[2]].tokens.append(new_token)

    else:
        # get token object to move
        tile_tokens = board_state[player_action[1]].tokens
        token = list(filter(lambda t: t.type.isupper() ==
                            (player_type == "upper"), tile_tokens))[0]

        # change token position
        token.update_position(player_action[2])

        # update board
        board_state[player_action[1]].tokens.remove(token)
        board_state[player_action[2]].tokens.append(token)

    if opponent_action[0] == "THROW":
        token_type = opponent_action[1] if player_type != "lower" else opponent_action[1].upper(
        )
        new_token = Token(token_type, opponent_action[2])

        # print(False, player_type, new_token.pos)
        opponent_tokens.append(new_token)
        # update game state
        board_state[opponent_action[2]].tokens.append(new_token)

    else:
        # get token object to move
        tile_tokens = board_state[opponent_action[1]].tokens
        token = list(filter(lambda t: t.type.isupper() ==
                            (player_type != "upper"), tile_tokens))[0]

        # change token position
        token.update_position(opponent_action[2])

        # update board
        board_state[opponent_action[1]].tokens.remove(token)
        board_state[opponent_action[2]].tokens.append(token)
    board_state = handle_collision(board_state, player_action[2])
    board_state = handle_collision(board_state, opponent_action[2])

    return board_state, player_tokens, opponent_tokens


def hex_distance(a, b):
    # form https://www.redblobgames.com/grids/hexagons/

    d = (abs(a[0] - b[0])
         + abs(a[0] + a[1] - b[0] - b[1])
         + abs(a[1] - b[1])) / 2

    return d if d else 0.1


def get_payoff(player_available_moves, opponent_available_moves, node, player_type):

    
    axis0 = len(player_available_moves) if len(player_available_moves) else 1
    axis1 = len(opponent_available_moves) if len(
        opponent_available_moves) else 1
    mat = np.zeros((axis0, axis1))

    board_state_mat = np.zeros((axis0, axis1)).tolist()
    player_tokens_mat = np.zeros((axis0, axis1)).tolist()
    opponent_tokens_mat = np.zeros((axis0, axis1)).tolist()
    player_remaining_tokens_mat = np.zeros((axis0, axis1)).tolist()
    opponent_remaining_tokens_mat = np.zeros((axis0, axis1)).tolist()

    
    s = time.time()
    for p_move in range(0, len(player_available_moves)):
        for o_move in range(0, len(opponent_available_moves)):
            
            player_remaining_tokens_mat[p_move][o_move] = node['player_remaining_tokens']
            if player_available_moves[p_move][0] == "THROW":
                 player_remaining_tokens_mat[p_move][o_move] -= 1

            opponent_remaining_tokens_mat[p_move][o_move] = node['opponent_remaining_tokens']
            if opponent_available_moves[o_move][0] == "THROW":
                opponent_remaining_tokens_mat[p_move][o_move] -= 1
            
            
            board_state_new, player_tokens_new, opponent_tokens_new = update_board_non_destructive(
                player_available_moves[p_move], opponent_available_moves[o_move], node['player_tokens'], node['opponent_tokens'], node['board_state'], player_type)
            
            
            mat[p_move][o_move] = score_move(board_state_new, player_type)
            

            board_state_mat[p_move][o_move] = board_state_new
            player_tokens_mat[p_move][o_move] = player_tokens_new
            opponent_tokens_mat[p_move][o_move] = opponent_tokens_new
    print('update_board_non_destructive: ', time.time() - s)       
            
    
    # print(mat)
    print('--------------', mat.shape)
    # probably_distribution = [0.5]*axis0
    s = time.time()
    probably_distribution = gametheory.solve_game(mat)[0]
    print('probably_distribution: ', time.time() - s)  
    
    # print(probably_distribution)

    board_state_list = []
    player_tokens_list = []
    opponent_tokens_list = []
    player_remaining_tokens_list = []
    opponent_remaining_tokens_list = []

    for p_move in range(0, len(player_available_moves)):
        max_score = 0
        max_i = 0
        for o_move in range(0, len(opponent_available_moves)):
            if mat[p_move][o_move] > max_score:
                max_score = mat[p_move][o_move]
                max_i = o_move
        board_state_list.append(board_state_mat[p_move][max_i])
        player_tokens_list.append(player_tokens_mat[p_move][max_i])
        opponent_tokens_list.append(opponent_tokens_mat[p_move][max_i])
        player_remaining_tokens_list.append(player_remaining_tokens_mat[p_move][max_i])
        opponent_remaining_tokens_list.append(opponent_remaining_tokens_mat[p_move][max_i])

    
    # print(list(range(0, len(player_available_moves))))
    # index = np.random.choice(
    #     a=range(0, len(player_available_moves)), size=1, p=probably_distribution)[0]

    return probably_distribution, player_tokens_list, opponent_tokens_list, board_state_list, player_remaining_tokens_list, opponent_remaining_tokens_list


node_prototype = {
    'score': 0,
    'my_score': 0,
    'explored': 0,
    'children': [],
    'parent': None,
    'player_tokens': None,
    'opponent_tokens': None,
    'board_state': None,
    'player_remaining_tokens': None,
    'opponent_remaining_tokens': None
}


def mcts(player_tokens, opponent_tokens, board_state, player_remaining_tokens, opponent_remaining_tokens, player_type):

    head = copy.deepcopy(node_prototype)
    head['player_tokens'] = player_tokens
    head['opponent_tokens'] = opponent_tokens
    head['board_state'] = board_state
    head['player_remaining_tokens'] = player_remaining_tokens
    head['opponent_remaining_tokens'] = opponent_remaining_tokens

    for i in range(1):
        # s = time.time()
        node = mcts_selection(head)
        # print('mcts_selection: ', time.time() - s)
        # s = time.time()
        nodes_added = mcts_expansion_simulation(node, player_type)
        # print('mcts_expansion_simulation: ', time.time() - s)

        if nodes_added == 0:
            break

    # pprint(head)

    max_score = 0
    best_node = 0

    for i in range(len(head['children'])):
        if head['children'][i]['score'] > max_score:
            max_score = head['children'][i]['score']
            best_node = i

    # for testing only remove in prod ------

    def remove_circular_refs(node):
        node.pop('parent')
        node.pop('board_state')
        node.pop('player_tokens')
        node.pop('opponent_tokens')
        node.pop('player_remaining_tokens')
        node.pop('opponent_remaining_tokens')
        node['name'] = "score: {:.2f}, explored: {}".format(
            node['score'], node['explored'])
        for i in node['children']:
            remove_circular_refs(i)

    remove_circular_refs(head)
    jsonString = json.dumps(head)
    # print(jsonString)

    jsonFile = open("data.json", "w")
    jsonFile.write(jsonString)

    # ------------------------------------

    return best_node


def mcts_selection(node):

    # UCB1 = V + C * sqrt(ln(N)/n)

    UCB1 = []

    node['explored'] += 1
    if len(node['children']) == 0:
        return node
    else:
        for i in node['children']:
            UCB1.append(
                i['score'] + UCB1_C * np.sqrt(
                    np.log(node['explored'])/i['explored']
                )
            )

        max = UCB1[0]
        max_i = 0
        for i in range(len(UCB1)):
            if UCB1[i] > max:
                max = UCB1[i]
                max_i = i

        # print(max_i)

        # print(UCB1)

        return mcts_selection(node['children'][max_i])


payoff = [0.9, 0.5, 0.1]


def mcts_expansion_simulation(node, player_type):
    if len(node['children']) == 0:
        # if len(payoff):
        #     payoff.pop()
        
        player_available_moves = get_available_moves(node['board_state'],
                                                     node['player_tokens'],
                                                     node['opponent_tokens'],
                                                     node['player_remaining_tokens'],
                                                     player_type)

        opponent_type = 'lower' if player_type == 'upper' else 'upper'
        opponent_available_moves = get_available_moves(node['board_state'],
                                                       node['opponent_tokens'],
                                                       node['player_tokens'],
                                                       node['opponent_remaining_tokens'],
                                                       opponent_type)
        
        probably_distribution, player_tokens_list, opponent_tokens_list, board_state_list, player_remaining_tokens_new, opponent_remaining_tokens_new = get_payoff(player_available_moves,
                                                                                                                                                                   opponent_available_moves, node, player_type)
        
        for i in range(0,len(probably_distribution)):
            new_node = copy.deepcopy(node_prototype)

            new_node['player_tokens'] = player_tokens_list[i]
            new_node['opponent_tokens'] = opponent_tokens_list[i]
            new_node['board_state'] = board_state_list[i]
            new_node['player_remaining_tokens'] = player_remaining_tokens_new[i]
            new_node['opponent_remaining_tokens'] = opponent_remaining_tokens_new[i]

            new_node['my_score'] = probably_distribution[i]
            new_node['score'] = probably_distribution[i]
            new_node['explored'] = 1
            new_node['parent'] = node
            node['children'].append(new_node)

        node['board_state'] = None
        node['player_available_moves'] = None
        node['opponent_available_moves'] = None
        node['player_remaining_tokens'] = None
        node['opponent_remaining_tokens'] = None

        mcts_update(node)
       
    return len(payoff)


def mcts_update(node):
    score = [node['my_score']]
    for child in node['children']:
        score.append(child['score'])
    node['score'] = np.average(score)
    if node['parent'] != None:
        mcts_update(node['parent'])


def score_move(board, player_type):

    token_list = {'R': [], 'P': [], 'S': [], 'r': [], 'p': [], 's': []}

    for pos, tile in board.items():
        if len(tile.tokens) > 0:
            for i in tile.tokens:
                # print(pos, i.type)
                token_list[i.type].append(pos)

    upper_score = 0
    lower_score = 0

    for i in ['R', 'P', 'S']:
        for j in token_list[i]:
            for k in token_list[TOKEN_TO_ATTACK[i.lower()]]:
                upper_score += 1/hex_distance(j, k)

            # # Check collisions with own token
            # for k in token_list[TOKEN_TO_ATTACK[i.lower()].upper()]:
            #     dist = hex_distance(j, k)
            #     if dist < 1:
            #         upper_score -= 1/hex_distance(j, k)

    for i in ['r', 'p', 's']:
        for j in token_list[i]:
            for k in token_list[TOKEN_TO_ATTACK[i].upper()]:
                lower_score += 1/hex_distance(j, k)

        # # Check collisions with own token
        # for k in token_list[TOKEN_TO_ATTACK[i]]:
        #         dist = hex_distance(j, k)
        #         if dist < 1:
        #             upper_score -= 1/hex_distance(j, k)

    # print('upper :', upper_score)
    # print('lower :', lower_score)

    score = upper_score - lower_score if player_type == 'upper' else lower_score - upper_score
    # print('score :', score)

    return score


head = {
    'score': 30,
    'my_score': 30,
    'children_score': [],
    'explored': 0,
    'children': [
    ],
    'parent': None
}

node = {
    'score': 10,
    'my_score': 10,
    'children_score': [],
    'explored': 0,
    'children': [
        {
            'score': 5,
            'my_score': 5,
            'explored': 0,
            'children': [],
            'parent': None
        },
        {
            'score': 2,
            'my_score': 2,
            'explored': 0,
            'children': [],
            'parent': None
        },
    ],
    'parent': head
}
# head['children'].append(node)

# print(head['score'])
# mcts_update(node)
# print(head['score'])

player_tokens = []
opponent_tokens = []
board_state = init_board(BOARD_SIZE)

import time
s = time.time()
print(mcts(player_tokens, opponent_tokens, board_state, 9, 9, 'upper'))
print(time.time()-s)