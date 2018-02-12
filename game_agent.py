
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

    return float((own_moves - opp_moves) + (distance_to_center_opponent - distance_to_center_player) * 0.634 / (game.move_count))


def custom_score_2(game, player):
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    y, x = game.get_player_location(player)
    boxes_to_center = abs(x - np.ceil(game.width/2)) + \
    abs(y - np.ceil(game.height/2)) - 1

    surface = game.width * game.height

    percentage_completed = game.move_count / surface
    
    if percentage_completed < 0.1:
        return 2 * own_moves - 0.5 * boxes_to_center
    
    elif percentage_completed > 40:
        return 2 * own_moves - opp_moves
    
    else:
        return float(3 * own_moves - opp_moves - 0.5 * boxes_to_center)


def custom_score_3(game, player):

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")


    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_moves_len = len(own_moves)
    opp_moves_len = len(opp_moves)

    w, h = game.get_player_location(game.get_opponent(player))
    y, x = game.get_player_location(player)
    
    distance_to_center_player = float((h - y)**2 + (w - x)**2)

    limit_positions = [(x, y) for x in (0, game.width-1) for y in range(game.width)] + \
    [(x, y) for y in (0, game.height-1) for x in range(game.height)]

    penalty = 0
    
    if (x, y) in limit_positions:
        penalty -= 1

    move_efficiency = 0

    for move in own_moves:
        y, x = move
        dist = float((h - y)**2 + (w - x)**2)
    
        if dist == 0:
            move_efficiency += 1
    
        else:
            move_efficiency += 1/dist
    
        if move in opp_moves:
            move_efficiency -= 1

    surface = game.width * game.height
    percentage_completed = game.move_count / surface

    if percentage_completed > 40:       
        return own_moves_len - 2 * opp_moves_len

    else:
        return float(own_moves_len - opp_moves_len - distance_to_center_player + move_efficiency + penalty)
    

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
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass

        return best_move

    def check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def stuck(self, game, depth):
        self.check_time()
        return (len(game.get_legal_moves()) == 0 and depth <= 0)

    def min_value(self, game, depth):
        self.check_time()
        
        if self.stuck(game, depth):
            return self.score(game, self)
        
        min_val = float("inf")
        legal_moves = game.get_legal_moves()
        
        for move in legal_moves:
            forecast = game.forecast_move(move)
            min_val = min(min_val, self.max_value(forecast, depth - 1))
        
        return min_val

    def max_value(self, game, depth):
        self.check_time()
        
        if self.stuck(game, depth):
            return self.score(game, self)
        
        max_val = float("-inf")
        legal_moves = game.get_legal_moves()
        
        for move in legal_moves:
            forecast = game.forecast_move(move)
            max_val = max(max_val, self.min_value(forecast, depth - 1))
        
        return max_val

    def minimax(self, game, depth):

        self.check_time()
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            return (-1, -1)
        
        vals = [(self.min_value(game.forecast_move(m), depth - 1), m) for m in legal_moves]
        _, move = max(vals)
        
        return move


class AlphaBetaPlayer(IsolationPlayer):

    def get_move(self, game, time_left):

        self.time_left = time_left
        legal_moves = game.get_legal_moves(self)

        if len(legal_moves) > 0:
            best_move = legal_moves[random.randint(0, len(legal_moves)-1)]
        
        else:
            best_move = (-1, -1)
        
        try:
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

    def check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def stuck(self, game, depth):
        self.check_time()
        return (len(game.get_legal_moves()) == 0 and depth <= 0)

    def min_value(self, game, depth, alpha, beta):
        self.check_time()
        best_move = (-1, -1)
        
        if self.stuck(game, depth):
            return (self.score(game, self), best_move)
        
        value = float("inf")
        legal_moves = game.get_legal_moves()
        
        for move in legal_moves:
            result = self.max_value(game.forecast_move(move), depth - 1, alpha, beta)
        
            if result[0] < value:
                value, _ = result
                best_move = move
        
            if value <= alpha:
                return (value, best_move)
        
            beta = min(beta, value)
        
        return (value, best_move)

    def max_value(self, game, depth, alpha, beta):
        self.check_time()
        best_move = (-1, -1)
        
        if self.stuck(game, depth):
            return (self.score(game, self), best_move)
        
        value = float("-inf")
        legal_moves = game.get_legal_moves()
        
        for move in legal_moves:
            result = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
        
            if result[0] > value:
                value, _ = result
                best_move = move
        
            if value >= beta:
                return (value, best_move)
        
            alpha = max(alpha, value)
        
        return (value, best_move)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        self.check_time()
        _, move = self.max_value(game, depth, alpha, beta)
        return move