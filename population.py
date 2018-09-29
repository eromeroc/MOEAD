#######################################################
# Autora: Elena Romero Contreras
# Clase Population
#######################################################

import numpy as np
import random

class Population:

	"""
	Clase que representa una población formada por vectores solución
	"""

	def __init__(self, pop_size, n_gens, upper_bound):
		"""
		Constructor de la clase Population
		Inicializa aleatoriamente la población de pop_size individuos
		con n_gens genes entre 0 y upper_bound
		"""
		self.pop_size = pop_size
		self.n_gens = n_gens
		self.population = []

		for i in range(pop_size):
			v_weight = []

			for j in range(n_gens):	
				v_weight.append(random.uniform(0,upper_bound))

			self.population.append(v_weight)


	def setIndiv(self, individual, index):
		"""
		Sustituye el individuo en la posición dada por index por el individuo dado
		"""
		self.population[index] = individual




