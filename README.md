# Machine Learning Homework 3

The following program was written to solve the pacman problem for a Technion Homework assignment.

Given the following, create an agent that plays an optimal strategy for Pacman.

- a Pacman board (inital board)
- the number of moves that Pacman will make

Unknowns:
- probability that the ghosts will move towards Pacman or move randomly
- the value of the 3 different dots (size = 1,2,3)

Our strategy for solving the problem

- Reinformencement Learning:
-- Every time we eat a 'dot' we record the value it earned us.
-- We use the expected value of each dot to improve our agents ability to choose which dot to collect.

While there is less than 5 values of each dot, we assume the expected value for the dot is the mean of values collected.
Once we have 5 values, we compare the SSE values of the data points fit to a list of distributions. The distribution
the smallest SSE is assumed to be the 'correctly guessed' distribution.

Then, we take the expected value of the distribution given our limiited data points.

-- Then, we calculate the possible given states in the next two rounds which assume the ghosts move towards pacman.

-- We compute the heuristic value for each state as follows =
