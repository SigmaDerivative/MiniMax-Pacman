# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from os import terminal_size
from util import manhattanDistance
from game import Directions
import random, util, time

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        value_and_action = self.minimax(game=gameState, depth=0)
        # print(f"minimax {value_and_action}")
        return value_and_action[1]

    def minimax(self, game, depth: int):
        """Main minimax search function, returns value and action"""
        # Using depth to iterate through all agents. Each actual depth is // num_agents
        terminal_state = depth == self.depth * game.getNumAgents()

        # Returns on terminated game to avoid asking for games without legal moves
        if game.isLose() or game.isWin() or terminal_state:
            return [self.evaluationFunction(game), None]

        # Get agent id based on depth
        agent_id = depth % game.getNumAgents()
        if agent_id == 0:
            # Pacman is 0
            return self.maximizer(game=game, depth=depth)
        else:
            return self.minimizer(game=game, depth=depth, ghost_id=agent_id)

    def maximizer(self, game, depth: int):
        """Maximizer for pacman"""
        # Start with worst value
        value_and_action = [float("-inf"), None]

        # Go through legal actions
        legal_actions = game.getLegalActions(0)
        for action in legal_actions:
            next_game = game.generateSuccessor(0, action)
            # Get the value of next game recursively
            next_value = self.minimax(game=next_game, depth=depth + 1)[0]

            # Update if better
            if next_value > value_and_action[0]:
                value_and_action = [next_value, action]

        return value_and_action

    def minimizer(self, game, depth: int, ghost_id: int):
        """Minimizer for ghosts"""
        value_and_action = [float("inf"), None]

        # Go through legal actions
        legal_actions = game.getLegalActions(ghost_id)
        for action in legal_actions:
            next_game = game.generateSuccessor(ghost_id, action)
            # Get the value of next game
            next_value = self.minimax(game=next_game, depth=depth + 1)[0]

            # Update if worse
            if next_value < value_and_action[0]:
                value_and_action = [next_value, action]

        return value_and_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value_and_action = self.alpha_beta(
            game=gameState, depth=0, alpha=float("-inf"), beta=float("inf")
        )
        # print(f"alphaBeta {value_and_action}")
        return value_and_action[1]

    def alpha_beta(self, game, depth: int, alpha: int, beta: int):
        """Main alpha-beta pruning search function, returns value and action"""
        # Using depth to iterate through all agents. Each actual depth is // num_agents
        number_of_agents = game.getNumAgents()
        terminal_state = depth == self.depth * number_of_agents

        # Returns on terminated game to avoid asking for games without legal moves
        if game.isLose() or game.isWin() or terminal_state:
            return [self.evaluationFunction(game), None]

        # Get agent id based on depth
        agent_id = depth % number_of_agents
        if agent_id == 0:
            # Pacman is 0
            return self.maximizer(game=game, depth=depth, alpha=alpha, beta=beta)
        else:
            return self.minimizer(
                game=game, depth=depth, ghost_id=agent_id, alpha=alpha, beta=beta
            )

    def maximizer(self, game, depth: int, alpha: int, beta: int):
        """Maximizer for pacman"""
        # Start with worst value
        value_and_action = [float("-inf"), None]

        # Go through legal actions
        legal_actions = game.getLegalActions(0)
        # if not legal_actions:
        #     return [self.evaluationFunction(game), None]
        for action in legal_actions:
            next_game = game.generateSuccessor(0, action)

            next_value = self.alpha_beta(
                game=next_game, depth=depth + 1, alpha=alpha, beta=beta
            )[0]

            # Update if better
            if next_value > value_and_action[0]:
                value_and_action = [next_value, action]
                # Update alpha
                alpha = max(value_and_action[0], alpha)

            # Prune
            if value_and_action[0] > beta:
                return value_and_action

        return value_and_action

    def minimizer(self, game, depth: int, ghost_id: int, alpha: int, beta: int):
        """Minimizer for ghosts"""
        value_and_action = [float("inf"), None]

        # Go through legal actions
        legal_actions = game.getLegalActions(ghost_id)
        # if not legal_actions:
        #     return [self.evaluationFunction(game), None]
        for action in legal_actions:
            next_game = game.generateSuccessor(ghost_id, action)

            next_value = self.alpha_beta(
                game=next_game, depth=depth + 1, alpha=alpha, beta=beta
            )[0]

            # Update if worse
            if next_value < value_and_action[0]:
                value_and_action = [next_value, action]
                # Update beta
                beta = min(value_and_action[0], beta)

            # Prune
            if value_and_action[0] < alpha:
                return value_and_action

        return value_and_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
