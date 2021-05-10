class Player:
    # ------------ GAME CONSTANTS ------------ #
    TOKEN_TO_ATTACK = {"r": "s", "p": "r", "s": "p"}
    INIT_TOKEN_COUNT = 9

    # ------------ GAME STATE ------------ #
    remaining_tokens = INIT_TOKEN_COUNT
    player_type = ""
    board_state = None
    player_tokens = []
    opponent_tokens = []

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
        self.board_state = Board()

    def action(self):
        """
        Called at the beginning of each turn. Based on the current state
        of the game, select an action to play this turn.
        """
        # put your code here

    def update(self, opponent_action, player_action):
        """
        Called at the end of each turn to inform this player of both
        players' chosen actions. Update your internal representation
        of the game state.
        The parameter opponent_action is the opponent's chosen action,
        and player_action is this instance's latest chosen action.
        """
        # put your code here

        update_board(player_action, opponent_action, board_state, player_tokens, opponent_tokens, player_type)


class Board:
    BOARD_SIZE = 4

    def __init__(self):
        self.board_dict = init_board(BOARD_SIZE)


class Tile:
    def __init__(self, r, q):
        self.r = r
        self.q = q
        self.pos = (r, q)
        self.occupying_tokens = []


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


def update_board(player_action, opponent_action, board_state, player_tokens, opponent_tokens, player_type):
    player_token_type = token_type if player_type == "lower" else token_type.upper()
    opponent_token_type = token_type.upper() if player_type == "lower" else token_type

    if player_action[0] == "THROW":
        new_token = Token(player_token_type, player_action[2])
        board_state[player_action[2]].occupying_tokens.append(new_token)
        player_tokens.append(new_token)
        
    if opponent_action[0] == "THROW":
        new_token = Token(opponent_token_type, opponent_action[2])
        board_state[opponent_action[2]].occupying_tokens.append(new_token)
        opponent_tokens.append(new_token)
