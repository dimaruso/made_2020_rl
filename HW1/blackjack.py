import gym
from gym import spaces
from gym.utils import seeding
from copy import deepcopy

class BlackjackEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        
        # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        
        # We play 3 decks
        self.deck3 = self.deck * 3 
        # self.deck3.sort()

        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        # hit: add a card to players hand and return
        if action == 1:  
            self.player.append(self.draw_card())
            if self.is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.

        # double: double reward, player draw one card, play out the dealers hand, and score
        elif action == 2:  
            done = True
            self.player.append(self.draw_card())
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = 2 * self.cmp(self.score(self.player), self.score(self.dealer))
        
        # stand: play out the dealers hand, and score
        else:  
            done = True
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = self.cmp(self.score(self.player), self.score(self.dealer))
            if self.natural and self.is_natural(self.player) and reward == 1.:
                reward = 1.5

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (self.sum_hand(self.player), self.dealer[0], \
                self.usable_ace(self.player))

    def _get_obs_with_deck(self): 
        """
        Counting all the cards turned out to be too difficult 
        and it seems that only Dustin Hoffman can do it. 
        We will calculate the sum and number of all cards.
        """
               
        # cards = deepcopy(self.deck3)
        # cards.append(self.dealer[1])  
        # cards.sort()

        sum_cards = sum(self.deck3) + self.dealer[1] # We take into account the dealer's card
        count_cards = len(self.deck3) + 1

        return (self.sum_hand(self.player), self.dealer[0], \
                self.usable_ace(self.player), sum_cards // count_cards)    

    def reset(self, state=None):
        if state is not None:
            self.player = state[0]                   
            self.dealer = state[1]
            self.deck3 = state[2]
        else:
            self.player = self.draw_hand()
            self.dealer = self.draw_hand()

        return self._get_obs()

    def get_state(self):
        return deepcopy([self.player, self.dealer, self.deck3])

    def render(self):
        print(f"player: {self.player}")
        print(f"dealer: {self.dealer}")
        print(f"usable_ace: {self.usable_ace(self.player)}")

    def draw_card(self):
        
        if len(self.deck3) <= 15: 
            self.deck3 = self.deck * 3
            # self.deck3.sort()
        
        card = self.np_random.choice(self.deck3)
        self.deck3.remove(card)

        return card

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def usable_ace(self, hand):  # Does this hand have a usable ace?
        return 1 in hand and sum(hand) + 10 <= 21

    def sum_hand(self, hand):  # Return current hand total
        if self.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    def is_bust(self, hand):  # Is this hand a bust?
        return self.sum_hand(hand) > 21

    def score(self, hand):  # What is the score of this hand (0 if bust)
        return 0 if self.is_bust(hand) else self.sum_hand(hand)

    def cmp(self, a, b):
        return float(a > b) - float(a < b)

    def is_natural(self, hand):  # Is this hand a natural blackjack?
        return sorted(hand) == [1, 10]
