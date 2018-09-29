#######################################################
# Author: Elena Romero Contreras
# Calculate fitness function
#######################################################

import data as dat
import numpy as np

L1 = 500
L2 = 500


def fitness(data,w):
	"""
	Método que calcula vector con las funciones objetivo intraclase e interclase
	"""
	w_penalty = sum([min(1,i) for i in w])

	return [data.dist_intra@w + L1*w_penalty, -data.dist_inter@w + L2*w_penalty]
