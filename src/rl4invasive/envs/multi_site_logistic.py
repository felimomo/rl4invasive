import gymnasium as gym
from gym import spaces
import numpy as np

from default_fns import default_utility

class LogisticMultiSite(gym.Env):
	"""
	Multi-site invasive population model. Population at N sites, which can exchange populations
	according to a matrix of weights. Some sites are also sources of population for the model,
	with a positive immigration rate. Each site has an independent recruitment rate given by
	the logistic equation.

	actions: N actions, 1 per site (intensity of removal)
	observations: N+1 observations, 1 pop obs per site plus month of the year

	METHODS:
		__init__
		reset
		step
		get_month
		pop_to_normpop
		normpop_to_pop
		observe

		check_config
	"""
	def __init__(
		self,
		config = {},
	):
		"""
		ARGS:
			config: dict
				'internal_imm': NxN matrix who's (ij)-th entry is the j->i immigration rate. (diagonal=0)
				'external_imm': N-vector who's i-th entry is the immigration rate to site i.
				'site_r': N-vector for low-pop recruitment rate r parameter at each site.
				'site_K': N-vector forcarrying capacity at each site.
				'seasonality': 12-vector encoding seasonality of recruitment
				'utility': callable, utility fn.
				'max_pop': float, maximum pop value allowed. (used for normalization)
				'init_pop': N-vector, initial pop
				'maxT': int, maximum episode length.

		ATTRS:
			pop: N-vector, lives in space [0, max_pop]
			normalized_pop: normalization of pop to [-1, +1]
			t: timestep
			month: t mod 12

			**elements of config
		"""

		#
		# dynamics
		self.internal_imm = config.get('internal_imm', np.eye(2)) 
		self.n_site = self.internal_imm.shape[0]
		self.external_imm = config.get('external_imm', np.zeros(2))
		self.site_r = config.get('site_r', 0.5 * np.ones(self.n_site))
		self.site_K = config.get('site_K', np.ones(self.n_site))
		self.seasonality = config.get(
			'seasonality',
			np.array([
				np.sin(np.pi * month / 12) ** 2
				for month in range(12)
			]) # TBD: I pulled this outta my ass, check on biology for more realism
		)
		#
		# other
		self.utility_fn = config.get('utility_fn', default_utility_fn)
		self.max_pop = config.get('max_pop', 10)
		self.init_pop = config.get('init_pop', 0.01 * np.ones(self.n_site))
		self.maxT = config.get('maxT', 1200) # 100 years in months
		#
		#
		self.observation_space = spaces.Box(
			(self.n_site+1) * np.float32([-1]),
			(self.n_site+1) * np.float32([1]),
			dtype=np.float32
		)
		self.action_space = spaces.Box(
			self.n_site * np.float32([-1]),
			self.n_site * np.float32([1]),
			dtype=np.float32
		)
		#
		#
		self.reset()

	def reset(self, seed=None, options=None):
		self.t = 0
		self.month = self.get_month()
		self.pop = self.init_pop
		observation = self.observe()
		info = {}
		return observation, info

	def step(self, action):
		removal_rates = self.get_removal_rates(action)
		self.month = self.get_month()
		self.new_pop = self.dynamics(self.t) - removal_rates
		reward = self.utility_fn(self.pop, self.new_pop, self.t, action)
		#
		self.pop = self.new_pop
		#
		observation = self.observe()
		terminated = False
		truncated = False
		info = {}
		#
		return observation, reward, terminated, truncated, info

	def observe(self):
		pop_obs = self.pop_to_normpop(self.pop) + 0.05 * np.random.normal()
		time_obs = self.get_month()
		raw_obs = np.float32([time_obs, *pop_obs])
		return np.clip(raw_obs, -1, 1)

	def dynamics(self, t):
		recruits_det = (
			self.seasonality[self.month] # seasonality of recr
			* self.site_r * self.pop # linear recr
			* (1 - self.pop / self.site_K) # carrying capacity
		)
		recruits = recruits_det * (1 + 0.1 * np.random.normal())
		pop_after_recr = self.pop + recruits
		#
		# internal immigration
		positive_term = self.internal_imm * self.pop # pop incoming at site
		negative_term = self.internal_imm.T * self.pop # pop leaving site (not super sure abt this one)
		net_internal_change = positive_term - negative_term
		#
		#
		new_pop = (
			pop_after_recr + net_internal_change + self.external_imm
		)
		#
		#
		return new_pop


	def pop_to_normpop(self, pop):
		return -1 + 2 * pop / self.max_pop

	def normpop_to_pop(self, norm_pop):
		return self.max_pop * (norm_pop + 1) / 2

	def get_removal_rates(self, action):
		efficiency = 0.5
		return efficiency * (action + 1) /2

	def get_month(self):
		""" gets month out of timestep """
		return self.t % 12

	def check_config(self):
		""" Checks that the config arg in __init__ is consistent/ok. """
		assert len(self.internal_imm.shape) == 2
		assert self.internal_imm.shape[0] == self.internal_imm.shape[1]
		assert len(self.external_imm) == self.n_site
		# etc




