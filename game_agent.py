
import random
import numpy as np
import isolation

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(game, player):

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    b, a = game.get_player_location(game.get_opponent(player))

    distance_to_center_player = float((h - y)**2 + (w - x)**2)
    distance_to_center_opponent = float((h - b)**2 + (w - a)**2)
    return float((own_moves - opp_moves) + (distance_to_center_opp - distance_to_center) * 0.634 / (game.move_count))


def custom_score_2(game, player):
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    pos_y, pos_x = game.get_player_location(player)
    number_of_boxes_to_center = abs(pos_x - np.ceil(game.width/2)) + \
    abs(pos_y - np.ceil(game.height/2)) - 1

    if amount_completed(0, 10, game):
        return 2*own_moves - 0.5*number_of_boxes_to_center
    elif amount_completed(10, 40, game):
        return float(3*own_moves - opp_moves - 0.5*number_of_boxes_to_center)
    else:
        return 2*own_moves - opp_moves


def custom_score_3(game, player):

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    own_moves = len(my_moves)
    opp_moves = len(opponent_moves)
    moves_so_far = 0
    for box in game._board_state:
        if box == 1:
            moves_so_far += 1    

    w, h = game.get_player_location(game.get_opponent(player))
    y, x = game.get_player_location(player)
    distance_to_center = float((h - y)**2 + (w - x)**2)

    wall_boxes = [(x, y) for x in (0, game.width-1) for y in range(game.width)] + \
    [(x, y) for y in (0, game.height-1) for x in range(game.height)]

    penalty = 0
    if (x, y) in wall_boxes:
        penalty -= 1

    quality_of_move = 0
    for move in my_moves:
        y, x = move
        dist = float((h - y)**2 + (w - x)**2)
        if dist == 0:
            quality_of_move += 1
        else:
            quality_of_move += 1/dist
        if move in opponent_moves:
            quality_of_move -= 1

    if amount_completed(0, 40, game):
        return float(own_moves - opp_moves - distance_to_center + quality_of_move + penalty)
    else:       
        return own_moves - 2*opp_moves

def amount_completed(low, high, game):

    percent = game.move_count/(game.height * game.width)*100
    if low <= percent and high > percent:
        return True
    return False

class IsolationPlayer:

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):

    def get_move(self, game, time_left):

        self.time_left = time_left
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def __min_value(self, game, depth):
        self.__check_time()
        if self.__is_terminal(game, depth):
            return self.score(game, self)
        min_val = float("inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            min_val = min(min_val, self.__max_value(forecast, depth - 1))
        return min_val

    def __max_value(self, game, depth):
        self.__check_time()
        if self.__is_terminal(game, depth):
            return self.score(game, self)
        max_val = float("-inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            max_val = max(max_val, self.__min_value(forecast, depth - 1))
        return max_val

    def __is_terminal(self, game, depth):
        """Helper method to check if we've reached the end of the game tree or
        if the maximum depth has been reached.
        """
        self.__check_time()
        if len(game.get_legal_moves()) != 0 and depth > 0:
            return False
        return True

    def __check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def minimax(self, game, depth):

        self.__check_time()
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        vals = [(self.__min_value(game.forecast_move(m), depth - 1), m) for m in legal_moves]
        _, move = max(vals)
        return move


class AlphaBetaPlayer(IsolationPlayer):

    def get_move(self, game, time_left):

        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves = game.get_legal_moves(self)
        if len(legal_moves) > 0:
            best_move = legal_moves[random.randint(0, len(legal_moves)-1)]
        else:
            best_move = (-1, -1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 1
            while True:
                current_move = self.alphabeta(game, depth)
                if current_move == (-1, -1):
                    return best_move
                else:
                    best_move = current_move
                depth += 1
        except SearchTimeout:
            return best_move
        return best_move

    def __max_value(self, game, depth, alpha, beta):
        self.__check_time()
        best_move = (-1, -1)
        if self.__is_terminal(game, depth):
            return (self.score(game, self), best_move)
        value = float("-inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            result = self.__min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if result[0] > value:
                value, _ = result
                best_move = move
            if value >= beta:
                return (value, best_move)
            alpha = max(alpha, value)
        return (value, best_move)

    def __min_value(self, game, depth, alpha, beta):
        self.__check_time()
        best_move = (-1, -1)
        if self.__is_terminal(game, depth):
            return (self.score(game, self), best_move)
        value = float("inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            result = self.__max_value(game.forecast_move(move), depth - 1, alpha, beta)
            if result[0] < value:
                value, _ = result
                best_move = move
            if value <= alpha:
                return (value, best_move)
            beta = min(beta, value)
        return (value, best_move)

    def __is_terminal(self, game, depth):
        self.__check_time()
        if len(game.get_legal_moves()) != 0 and depth > 0:
            return False
        return True

    def __check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        self.__check_time()
        _, move = self.__max_value(game, depth, alpha, beta)
        return move