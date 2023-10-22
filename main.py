# Markov Reward process
# 22/10/2023
# Simple Python code for simulating a MRP <S,P,R,gamma>
import random
import numpy as np


class RandomWalkSM:
    """
    A class to represent a random walk environment.

    ...

    Attributes
    ----------
    start : int
        location of the agent at the beginning
    world : list of 2 element
        boundries of our environment
    gamma : float
        discount factor

    Methods
    -------
    step():
        used for taking a step inside our world.
   bellman():
        used to solve bellman eqn
   compute_p():
        used to compute transition probability matrix
    """
    def __init__(self, start, world, gamma):
        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
    start : int
        location of the agent at the beginning
    world : list of 2 element
        boundries of our environment
    gamma : float
        discount factor
        """
        if start > world[1] or start < world[0]:
            raise Exception("Invalid Initialization of SM")
        else:
            self.start = start
            self.world = world
            self.pos = start
            self.gamma = gamma
            self.steps = 0
            self.state = "neutral"
            self.reward = 0
            self.V = np.zeros((world[1] - world[0] + 1, 1), float)
            self.P = np.zeros((world[1] - world[0] + 1, world[1] - world[0] + 1), float)
            self.compute_p(world)

    def step(self):
        '''
        used for taking a step inside our world

                Parameters:
                        N/A

                Returns:
                        N/A
        '''
        self.pos += random.randint(-1, 1)
        self.steps += 1
        self.reward -= 1
        if self.pos == self.world[0]:
            self.state = "lost"
        elif self.pos == self.world[1]:
            self.state = "win"

    def bellmam(self):
        '''
        used to solve bellman eqn

                Parameters:
                        N/A

                Returns:
                        N/A
        '''

        # closed form solution
        r = -1.0 * np.ones((self.world[1] - self.world[0] + 1, 1))
        r[0][0] = 0
        r[self.world[1] - self.world[0]][0] = 0
        self.V = np.matmul(np.linalg.inv(np.identity(self.world[1] - self.world[0] + 1) - self.gamma * self.P),
                           r)

    def compute_p(self, size):
        '''
        used to compute transition probability matrix

                Parameters:
                        N/A

                Returns:
                        N/A
        '''
        for i in range(0, size[1] - size[0] + 1):
            for j in range(0, size[1] - size[0] + 1):
                if (i == 0 or i == size[1] - size[0]):
                    self.P[i][j] = 0
                elif (i == j or i == j - 1 or i == j + 1):
                    self.P[i][j] = 1 / 3


if __name__ == '__main__':
    W1 = RandomWalkSM(50, [0, 100], 0.9)
    while W1.state == 'neutral':
        W1.step()
    print(W1.steps)
    print(W1.state)
    print(W1.reward)
    print(W1.P)
    W1.bellmam()
    print(W1.V)
