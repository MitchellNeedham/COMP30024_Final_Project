# COMP300024 Final Project
# Authors:
#  - Mitchell Needham  : 823604
#  - Ed Hinrichsen     : 993558

import time
import copy
import numpy as np
# import json
import scipy.optimize as opt


# ------------ GAME CONSTANTS ------------ #
TOKEN_TO_ATTACK = {"r": "s", "p": "r", "s": "p"}
TOKEN_TYPES = ["r", "p", "s"]
INIT_TOKEN_COUNT = 9
BOARD_SIZE = 4
UCB1_C = 2
reward_penitential = 10
LARGE_NUM = 999999


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

        self.player_available_moves = get_available_moves(self.board_state,
                                                          self.player_tokens,
                                                          self.opponent_tokens,
                                                          self.player_remaining_tokens,
                                                          self.player_type)
        index = mcts(self.player_tokens, self.opponent_tokens, self.board_state, self.player_remaining_tokens,
                     self.opponent_remaining_tokens, self.player_type)
        # return get_payoff(self.player_available_moves, self.opponent_available_moves, self.board_state, self.player_type)
        return self.player_available_moves[index]

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
    board_state, player_tokens, opponent_tokens = handle_collision(board_state, player_action[2], player_tokens,
                                                                   opponent_tokens, player_type)
    board_state, player_tokens, opponent_tokens = handle_collision(board_state, opponent_action[2], player_tokens,
                                                                   opponent_tokens, player_type)

    return board_state, player_tokens, opponent_tokens

def handle_collision(board_state, pos, player_tokens, opponent_tokens, player_type):
    tokens = board_state[pos].tokens
    if len(tokens) < 2:
        return board_state, player_tokens, opponent_tokens

    to_destroy = []

    for attacker_token in tokens.copy():
        for attacked_token in tokens.copy():
            can_attack = (TOKEN_TO_ATTACK[attacker_token.type.lower()] == attacked_token.type.lower())
            if can_attack:
                to_destroy.append(attacked_token)

    for token in set(to_destroy):
        if token.type.isupper() == (player_type == "upper"):
            player_tokens.remove(token)
        else:
            opponent_tokens.remove(token)

        board_state[pos].tokens.remove(token)

    return board_state, player_tokens, opponent_tokens

def valid_path(r, q):
    if r > BOARD_SIZE or q > BOARD_SIZE or r < -BOARD_SIZE or q < -BOARD_SIZE or abs(r + q) > BOARD_SIZE:
        return None
    return r, q

def get_available_moves(board_state, player_tokens, opponent_tokens, remaining_tokens, player_type):
    available_moves = []
    collision_locations = map(lambda x: x.pos, opponent_tokens)

    for token in player_tokens:
        neighbour_tiles = filter(None, board_state[token.pos].neighbours)
        swing_locations = filter(None, get_swing_locations(board_state, token))

        safe_neighbours = list(set(neighbour_tiles).difference(collision_locations))
        safe_swing_locations = list(
            set(swing_locations).difference(neighbour_tiles).difference(collision_locations))

        available_moves += map(lambda x: ("SLIDE", token.pos, x), safe_neighbours)
        available_moves += map(lambda x: ("SWING", token.pos, x), safe_swing_locations)

    available_moves += get_available_throws(board_state, remaining_tokens, player_type)

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
    neighbours = board_state[token.pos].neighbours
    for tile in filter(None, neighbours):
        for neighbour_token in board_state[tile].tokens:
            if neighbour_token.type.isupper() != token.type.isupper():
                continue
            swingable_tiles = list(
                set(board_state[neighbour_token.pos].neighbours).difference(
                    board_state[token.pos].neighbours + [token.pos]))
            # print(token.pos, swingable_tiles, board_state[token.pos].neighbours)
            for swing_tile in swingable_tiles:
                if swing_tile in neighbours or swing_tile == token.pos:
                    continue
                swing_locations.append(swing_tile)
    return list(set(swing_locations))


def update_board_non_destructive(player_action, opponent_action, player_tokens, opponent_tokens, board_state,
                                 player_type, undo=False):

    player_token = None
    opponent_token = None
    if player_action:
        if undo:
            # print('undo')
            player_action = list(player_action)
            if player_action[0] == "THROW":
                player_action[0] = "REMOVE"
            else:
                swap = player_action[1]
                player_action[1] = player_action[2]
                player_action[2] = swap

        if player_action[0] == "THROW":
            token_type = player_action[1] if player_type == "lower" else player_action[1].upper()
            new_token = Token(token_type, player_action[2])

            player_tokens.append(new_token)
            # update game state
            board_state[player_action[2]].tokens.append(new_token)

            player_token = new_token

        elif player_action[0] == "REMOVE":
            tile_tokens = board_state[player_action[2]].tokens  
            token = list(filter(lambda t: t.type.isupper() ==
                                          (player_type == "upper"), tile_tokens))[0]
            if token in player_tokens:
                player_tokens.remove(token)
            board_state[player_action[2]].tokens.remove(token)

        else:
            if len(board_state[player_action[1]].tokens) > 0:
                # get token object to move
                tile_tokens = board_state[player_action[1]].tokens
                token = list(filter(lambda t: t.type.isupper() ==
                                            (player_type == "upper"), tile_tokens))
                if len(token) > 0:
                    token = token[0]
                # change token position
                    token.update_position(player_action[2])

                    # update board
                    board_state[player_action[1]].tokens.remove(token)
                    board_state[player_action[2]].tokens.append(token)

                    player_token = token

        if player_token:
            player_token = (player_token.type, player_token.pos)

    if opponent_action:
        if undo:
            opponent_action = list(opponent_action)
            if opponent_action[0] == "THROW":
                opponent_action[0] = "REMOVE"
            else:
                swap = opponent_action[1]
                opponent_action[1] = opponent_action[2]
                opponent_action[2] = swap

        if opponent_action[0] == "THROW":
            token_type = opponent_action[1] if player_type != "lower" else opponent_action[1].upper()
            new_token = Token(token_type, opponent_action[2])

            opponent_tokens.append(new_token)
            board_state[opponent_action[2]].tokens.append(new_token)

            opponent_token = new_token

        elif opponent_action[0] == "REMOVE":
            tile_tokens = board_state[opponent_action[2]].tokens
            token = list(filter(lambda t: t.type.isupper() ==
                                          (player_type != "upper"), tile_tokens))[0]
            if token in opponent_tokens:
                opponent_tokens.remove(token)
            board_state[opponent_action[2]].tokens.remove(token)

        else:
            # get token object to move
            if len(board_state[opponent_action[1]].tokens) > 0:
                tile_tokens = board_state[opponent_action[1]].tokens
                token = list(filter(lambda t: t.type.isupper() ==
                                            (player_type != "upper"), tile_tokens))
                if len(token) > 0:
                    token = token[0]

                # change token position
                    token.update_position(opponent_action[2])

                    # update board
                    board_state[opponent_action[1]].tokens.remove(token)
                    board_state[opponent_action[2]].tokens.append(token)

                    opponent_token = token

        if opponent_token:
            opponent_token = (opponent_token.type, opponent_token.pos)

    return board_state, player_token, opponent_token


def hex_distance(a, b):
    # form https://www.redblobgames.com/grids/hexagons/

    d = (abs(a[0] - b[0])
         + abs(a[0] + a[1] - b[0] - b[1])
         + abs(a[1] - b[1])) / 2

    return d if d else (1 / reward_penitential)


def get_payoff(player_available_moves, opponent_available_moves, player_tokens, opponent_tokens, board_state, node,
               player_type):
    axis0 = len(player_available_moves) if len(player_available_moves) else 1
    axis1 = len(opponent_available_moves) if len(
        opponent_available_moves) else 1
    mat = np.zeros((axis0, axis1))

    player_actions_mat = np.zeros((axis0, axis1)).tolist()
    opponent_action_mat = np.zeros((axis0, axis1)).tolist()
    player_remaining_tokens_mat = np.zeros((axis0, axis1)).tolist()
    opponent_remaining_tokens_mat = np.zeros((axis0, axis1)).tolist()

    for p_move in range(0, len(player_available_moves)):
        for o_move in range(0, len(opponent_available_moves)):

            player_remaining_tokens_mat[p_move][o_move] = node['player_remaining_tokens']
            if player_available_moves[p_move][0] == "THROW":
                player_remaining_tokens_mat[p_move][o_move] -= 1

            opponent_remaining_tokens_mat[p_move][o_move] = node['opponent_remaining_tokens']
            if opponent_available_moves[o_move][0] == "THROW":
                opponent_remaining_tokens_mat[p_move][o_move] -= 1

            # print(len(player_tokens))
            board_state_new, player_move, opponent_move = update_board_non_destructive(
                player_available_moves[p_move], opponent_available_moves[o_move], player_tokens, opponent_tokens,
                board_state, player_type)

            mat[p_move][o_move] = score_move(player_move, opponent_move, board_state_new, player_type)

            update_board_non_destructive(
                player_available_moves[p_move], opponent_available_moves[o_move], player_tokens, opponent_tokens,
                board_state, player_type, undo=True)

            player_actions_mat[p_move][o_move] = player_available_moves[p_move]
            opponent_action_mat[p_move][o_move] = opponent_available_moves[o_move]

    probably_distribution = solve_game(mat)[0]

    player_actions_list = []
    opponent_actions_list = []
    player_remaining_tokens_list = []
    opponent_remaining_tokens_list = []

    for p_move in range(0, len(player_available_moves)):
        min_score = LARGE_NUM
        min_i = 0
        for o_move in range(0, len(opponent_available_moves)):
            if mat[p_move][o_move] < min_score:
                min_score = mat[p_move][o_move]
                min_i = o_move
        # print(min_i)
        player_actions_list.append(player_actions_mat[p_move][min_i])
        opponent_actions_list.append(opponent_action_mat[p_move][min_i])
        player_remaining_tokens_list.append(player_remaining_tokens_mat[p_move][min_i])
        opponent_remaining_tokens_list.append(opponent_remaining_tokens_mat[p_move][min_i])

    return probably_distribution, player_actions_list, opponent_actions_list, player_remaining_tokens_list, opponent_remaining_tokens_list


node_prototype = {
    'score': 0,
    'my_score': 0,
    'explored': 0,
    'children': [],
    'parent': None,
    'player_action': None,
    'opponent_action': None,
    'player_remaining_tokens': None,
    'opponent_remaining_tokens': None
}


def mcts(player_tokens, opponent_tokens, board_state, player_remaining_tokens, opponent_remaining_tokens, player_type):
    head = copy.deepcopy(node_prototype)

    head['player_action'] = None
    head['opponent_action'] = None
    head['player_remaining_tokens'] = player_remaining_tokens
    head['opponent_remaining_tokens'] = opponent_remaining_tokens

    start_time = time.time()
    calls = 0

    while time.time() - start_time < 0.7:
        calls += 1
        player_tokens_new = copy.deepcopy(player_tokens)
        opponent_tokens_new = copy.deepcopy(opponent_tokens)
        board_state_new = copy.deepcopy(board_state)
        node = mcts_selection(head, player_tokens_new, opponent_tokens_new, board_state_new, player_type)

        nodes_added = mcts_expansion_simulation(node, player_tokens_new, opponent_tokens_new, board_state_new,
                                                player_type)

        if nodes_added == 0:
           break

    probably_distribution = []

    for i in range(len(head['children'])):
        val = head['children'][i]['score']
        probably_distribution.append(val)

    total = sum(probably_distribution)

    for i in range(len(probably_distribution)):
        probably_distribution[i] = probably_distribution[i] / total
    index = np.random.choice(a=range(0, len(probably_distribution)), size=1, p=probably_distribution)[0]

    # # for testing only remove in prod ------

    # def remove_circular_refs(node):
    #     node.pop('parent')
    #     # node.pop('player_action')
    #     # node.pop('opponent_action')
    #     node.pop('player_remaining_tokens')
    #     node.pop('opponent_remaining_tokens')
    #     node['name'] = "s: {:.2f}, e: {}\n".format(
    #         node['score'], node['explored']) + str(node['player_action']) + " " + str(node['opponent_action'])
    #     for i in node['children']:
    #         remove_circular_refs(i)

    # remove_circular_refs(head)
    # jsonString = json.dumps(head)
    # # print(jsonString)

    # jsonFile = open("data.json", "w")
    # jsonFile.write(jsonString)

    # # ------------------------------------
    print("calls: ", calls)
    return index


def mcts_selection(node, player_tokens, opponent_tokens, board_state, player_type):
    player_action = node['player_action']
    opponent_action = node['opponent_action']
    update_board_non_destructive(player_action, opponent_action, player_tokens, opponent_tokens, board_state,
                                 player_type)
    UCB1 = []

    node['explored'] += 1
    if len(node['children']) == 0:
        return node
    else:
        for i in node['children']:
            UCB1.append(
                i['score'] + UCB1_C * np.sqrt(
                    np.log(node['explored']) / i['explored']
                )
            )

        max = UCB1[0]
        max_i = 0
        for i in range(len(UCB1)):
            if UCB1[i] > max:
                max = UCB1[i]
                max_i = i

        return mcts_selection(node['children'][max_i], player_tokens, opponent_tokens, board_state, player_type)


def mcts_expansion_simulation(node, player_tokens, opponent_tokens, board_state, player_type):
    if len(node['children']) == 0:

        player_available_moves = get_available_moves(board_state,
                                                     player_tokens,
                                                     opponent_tokens,
                                                     node['player_remaining_tokens'],
                                                     player_type)
        opponent_type = 'lower' if player_type == 'upper' else 'upper'
        opponent_available_moves = get_available_moves(board_state,
                                                       opponent_tokens,
                                                       player_tokens,
                                                       node['opponent_remaining_tokens'],
                                                       opponent_type)

        probably_distribution, player_actions_list, opponent_actions_list, player_remaining_tokens_list, opponent_remaining_tokens_list = get_payoff(
            player_available_moves,
            opponent_available_moves, player_tokens, opponent_tokens, board_state, node, player_type)
        for i in range(0, len(probably_distribution)):
            new_node = copy.deepcopy(node_prototype)

            new_node['player_action'] = player_actions_list[i]
            new_node['opponent_action'] = opponent_actions_list[i]
            new_node['player_remaining_tokens'] = player_remaining_tokens_list[i]
            new_node['opponent_remaining_tokens'] = opponent_remaining_tokens_list[i]

            new_node['my_score'] = probably_distribution[i]
            new_node['score'] = probably_distribution[i]
            new_node['explored'] = 1
            new_node['parent'] = node
            node['children'].append(new_node)

        mcts_update(node)

    return len(probably_distribution)


def mcts_update(node):
    score = [node['my_score']]
    find_max = []
    for child in node['children']:
        find_max.append(child['score'])
    score.append(max(find_max))
    node['score'] = np.average(score)
    if node['parent'] is not None:
        mcts_update(node['parent'])

# Original by Matthew Farrugia-Roberts, 2021
def solve_game(V, maximiser=True, rowplayer=True):
    """
    Given a utility matrix V for a zero-sum game, compute a mixed-strategy
    security strategy/Nash equilibrium solution along with the bound on the
    expected value of the game to the player.
    By default, assume the player is the MAXIMISER and chooses the ROW of V,
    and the opponent is the MINIMISER choosing the COLUMN. Use the flags to
    change this behaviour.

    Parameters
    ----------
    * V: (n, m)-array or array-like; utility/payoff matrix;
    * maximiser: bool (default True); compute strategy for the maximiser.
        Set False to play as the minimiser.
    * rowplayer: bool (default True); compute strategy for the row-chooser.
        Set False to play as the column-chooser.

    Returns
    -------
    * s: (n,)-array; probability vector; an equilibrium mixed strategy over
        the rows (or columns) ensuring expected value v.
    * v: float; mixed security level / guaranteed minimum (or maximum)
        expected value of the equilibrium mixed strategy.

    Exceptions
    ----------
    * OptimisationError: If the optimisation reports failure. The message
        from the optimiser will accompany this exception.
    """
    V = np.asarray(V)
    # lprog will solve for the column-maximiser
    if rowplayer:
        V = V.T
    if not maximiser:
        V = -V
    m, n = V.shape
    # ensure positive
    c = -V.min() + 1
    Vpos = V + c
    # solve linear program
    res = opt.linprog(
        np.ones(n),
        A_ub=-Vpos,
        b_ub=-np.ones(m),
    )
    if res.status:
        raise OptimisationError(res.message)  # TODO: propagate whole result
    # compute strategy and value
    v = 1 / res.x.sum()
    s = res.x * v
    v = v - c  # re-scale
    if not maximiser:
        v = -v
    return s, v

# Original by Matthew Farrugia-Roberts, 2021
class OptimisationError(Exception):
    """For if the optimiser reports failure."""


def score_move(player_token, opponent_token, board, player_type):

    if player_token == None or opponent_token == None:
        return 0

    token_list = {'R': [], 'P': [], 'S': [], 'r': [], 'p': [], 's': []}

    for pos, tile in board.items():
        if len(tile.tokens) > 0:
            for i in tile.tokens:
                # print(pos, i.type)
                token_list[i.type].append(pos)

    player_score = 0
    opponent_score = 0

    token_type = player_token[0]
    token_pos = player_token[1]
    tta = TOKEN_TO_ATTACK[token_type.lower()].upper() if player_type == 'lower' else TOKEN_TO_ATTACK[token_type.lower()]
    for k in token_list[tta]:
        if len(token_list[tta]) > 0:
            player_score += 1 / hex_distance(token_pos, k)
            
    for i in board[token_pos].tokens:
        if token_type.lower() == TOKEN_TO_ATTACK[i.type.lower()]:
            player_score -= reward_penitential*10
        if player_type == 'upper' and TOKEN_TO_ATTACK[token_type.lower()].upper() == i.type:
            player_score -= reward_penitential*2
        elif player_type == 'lower' and TOKEN_TO_ATTACK[token_type.lower()] == i.type:
            player_score -= reward_penitential*2

    token_type = opponent_token[0]
    token_pos = opponent_token[1]
    tta = TOKEN_TO_ATTACK[token_type.lower()].upper() if player_type == 'lower' else TOKEN_TO_ATTACK[token_type.lower()]
    for k in token_list[tta]:
        if len(token_list[tta]) > 0:
            opponent_score += 1 / hex_distance(token_pos, k)

    for i in board[token_pos].tokens:
        if token_type.lower() == TOKEN_TO_ATTACK[i.type.lower()]:
            opponent_score -= reward_penitential*10
        if player_type == 'upper' and TOKEN_TO_ATTACK[token_type.lower()].upper() == i.type:
            opponent_score -= reward_penitential*2
        elif player_type == 'lower' and TOKEN_TO_ATTACK[token_type.lower()] == i.type:
            opponent_score -= reward_penitential

    return player_score - opponent_score
