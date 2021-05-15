# COMP300024 Final Project
# Authors:
#  - Mitchell Needham  : 823604
#  - Ed Hinrichsen     : XXXXXX

import random

# ------------ GAME CONSTANTS ------------ #
TOKEN_TO_ATTACK = {"r": "s", "p": "r", "s": "p"}
TOKEN_TYPES = ["r", "p", "s"]
INIT_TOKEN_COUNT = 9
BOARD_SIZE = 4


class Player:
    # ------------ GAME STATE ------------ #
    remaining_tokens = INIT_TOKEN_COUNT
    player_type = ""
    board_state = None
    player_tokens = list()
    opponent_tokens = list()

    # ------------ MOVE DATA ------------ #
    available_moves = []

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
        self.board_state = init_board(BOARD_SIZE)
        self.available_moves = get_available_throws(self.board_state, self.remaining_tokens, player)

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        # put your code here
        return random.choice(self.available_moves)

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        # put your code here

        self.available_moves = []

        if player_action[0] == "THROW":
            self.remaining_tokens -= 1

        self.board_state, self.player_tokens, self.opponent_tokens = \
            update_board(player_action,
                         opponent_action,
                         self.player_tokens,
                         self.opponent_tokens,
                         self.board_state,
                         self.player_type)

        self.available_moves = get_available_moves(self.board_state,
                                                   self.player_tokens,
                                                   self.opponent_tokens,
                                                   self.remaining_tokens,
                                                   self.player_type)


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
        token_type = player_action[1] if player_type == "lower" else player_action[1].upper()
        new_token = Token(token_type, player_action[2])

        player_tokens.append(new_token)
        # update game state
        board_state[player_action[2]].tokens.append(new_token)

    else:
        # get token object to move
        tile_tokens = board_state[player_action[1]].tokens
        token = list(filter(lambda t: t.type.isupper() == (player_type == "upper"), tile_tokens))[0]

        # change token position
        token.update_position(player_action[2])

        # update board
        board_state[player_action[1]].tokens.remove(token)
        board_state[player_action[2]].tokens.append(token)

    if opponent_action[0] == "THROW":
        token_type = opponent_action[1] if player_type != "lower" else opponent_action[1].upper()
        new_token = Token(token_type, opponent_action[2])

        opponent_tokens.append(new_token)
        # update game state
        board_state[opponent_action[2]].tokens.append(new_token)

    else:
        # get token object to move
        tile_tokens = board_state[opponent_action[1]].tokens
        token = list(filter(lambda t: t.type.isupper() == (player_type != "upper"), tile_tokens))[0]

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
        safe_swing_locations = list(set(swing_locations).difference(neighbour_tiles).difference(collision_locations))

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
            for swing_tile in swingable_tiles:
                if swing_tile in neighbours or swing_tile == token.pos:
                    continue
                swing_locations.append(swing_tile)
    return list(set(swing_locations))
