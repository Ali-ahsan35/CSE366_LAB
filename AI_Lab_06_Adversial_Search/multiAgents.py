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


from util import manhattanDistance
from game import Directions
import random, util

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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

    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        """
        def minimax(agentIndex, depth, state):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                return max(minimax(1, depth, state.generateSuccessor(agentIndex, action))
                           for action in state.getLegalActions(agentIndex))
            else:  # Ghosts' turn (minimizing player)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                return min(minimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action))
                           for action in state.getLegalActions(agentIndex))

        actions = gameState.getLegalActions(0)
        bestAction = max(actions, key=lambda action: minimax(1, 0, gameState.generateSuccessor(0, action)))
        return bestAction
  class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(agentIndex, depth, state, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    value = max(value, alphaBeta(1, depth, state.generateSuccessor(agentIndex, action), alpha, beta))
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value
            else:  # Ghosts' turn
                value = float('inf')
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    value = min(value, alphaBeta(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action), alpha, beta))
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                return value

        actions = gameState.getLegalActions(0)
        bestAction = max(actions, key=lambda action: alphaBeta(1, 0, gameState.generateSuccessor(0, action), float('-inf'), float('inf')))
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
        """
        def expectimax(agentIndex, depth, state):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                return max(expectimax(1, depth, state.generateSuccessor(agentIndex, action))
                           for action in state.getLegalActions(agentIndex))
            else:  # Ghosts' turn (chance nodes)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                actions = state.getLegalActions(agentIndex)
                probabilities = [1 / len(actions)] * len(actions)
                return sum(probabilities[i] * expectimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action))
                           for i, action in enumerate(actions))

        actions = gameState.getLegalActions(0)
        bestAction = max(actions, key=lambda action: expectimax(1, 0, gameState.generateSuccessor(0, action)))
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    foodDistances = [manhattanDistance(position, foodPos) for foodPos in food.asList()]
    closestFoodDistance = min(foodDistances) if foodDistances else 0

    ghostDistances = [manhattanDistance(position, ghost.getPosition()) for ghost in ghostStates]
    closestGhostDistance = min(ghostDistances) if ghostDistances else float('inf')

    ghostPenalty = -200 if closestGhostDistance <= 1 and not any(scaredTimes) else 0
    foodReward = -1 * closestFoodDistance

    return currentGameState.getScore() + foodReward + ghostPenalty + sum(scaredTimes)


# Abbreviation
better = betterEvaluationFunction

