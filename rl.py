import numpy as np
import lspi
from lspi.domains import Domain, ChainDomain
from lspi.solvers import LSTDQSolver
from lspi.policy import Policy
from lspi.sample import Sample
from lspi.basis_functions import FakeBasis, OneDimensionalPolynomialBasis

if __name__ == '__main__':
  # data = [
  #   Sample(np.array([0]), 0, 1, np.array([0])),
  #   Sample(np.array([1]), 0, -1, np.array([1]), True)
  # ]

  precondition_value = .3
  initial_policy = Policy(OneDimensionalPolynomialBasis(3,2), .9, 0, tie_breaking_strategy=Policy.TieBreakingStrategy.FirstWins)
  # initial_policy = Policy(lspi.basis_functions.RadialBasisFunction(np.array([[0], [2], [4], [6], [8]]), .5, 2), .9, 0)
  sampling_policy = Policy(FakeBasis(2), .9, 1)
  solver = LSTDQSolver(precondition_value)
  # weights = solver.solve(data[:-1], initial_policy)
  domain = ChainDomain()
  samples = []

  for i in range(1000):
    action = sampling_policy.select_action(domain.current_state())
    samples.append(domain.apply_action(action))

  learned_policy = lspi.learn(samples, initial_policy, solver)

  domain.reset()

  cumulative_reward = 0

  for i in range(1000):
    action = learned_policy.best_action(domain.current_state())
    sample = domain.apply_action(action)
    print action
    cumulative_reward += sample.reward

  print cumulative_reward
