#######################################################
# Autora: Elena Romero Contreras
# Implementación del algoritmo MOEAD 
#######################################################

import math
import random
import numpy as np
import population as pop
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def MOEAD(data=None, pop_size=100, fitness=None, CR=0.95, F=0.7, MAX_EVAL=100, gamma=0.1, upper_bound=1):
	"""
	Algoritmo evolutivo multiobjetivo basado en descomposición
	Parámetros:
		data: conjunto de datos
		pop_size: tamaño de la población
		fitness: función multiobjetivo a evaluar
		CR: probabilidad de cruce
		F: factor de escala
		MAX_EVAL: nº máximo de iteraciones.
		gamma: valor umbral a partir del cual una característica se considera 0
		upper_bound: límite superior del espacio de búsqueda
	"""
	n_eval = 1
	n_neighbors = 20
	# Inicializamos aleatoriamente población
	population = pop.Population(pop_size = pop_size, n_gens = data.n_col, upper_bound = upper_bound)
	child = [k[:] for k in population.population[:]]
	#Inicializamos punto ideal
	ideal_point = [math.inf, math.inf]

    # Evaluamos la población inicial y actualizamos el punto ideal
	parent_fit = [fitness(data, population.population[i]) for i in range(population.pop_size)]
	child_fit = [k[:] for k in parent_fit[:]]

	ideal_point = [min(min(np.transpose(parent_fit)[i]),ideal_point[i]) for i in range(2)]

	# Inicializamos pesos para las funciones objetivo
	weight = np.zeros((population.pop_size, 2))

	for i in range(population.pop_size):
		weight[i][0] = i/(population.pop_size-1)
		weight[i][1] = 1-i/(population.pop_size-1)


	# Calculamos la matriz de distancias y los vecinos de los vectores de pesos del MOEA/D
	distance = np.zeros((population.pop_size, population.pop_size))	# Matriz de distancias entre pesos
	neighbour_index = np.zeros((population.pop_size, n_neighbors))	# Indices de los vecinos de cada vector weight

	for i in range(population.pop_size):
		for j in range(population.pop_size):
			distance[i][j] = np.linalg.norm(weight[i,:] - weight[j,:])
		#Ordeno las distancias y me quedo con los índices	
		indexes = np.argsort(distance[i,:])
		#Me quedo con los más cercanos
		neighbour_index[i,:]=indexes[0:n_neighbors]

	while(n_eval < MAX_EVAL):

		for i in range(population.pop_size):
			child[i] = diffEvolution(data, population.population, neighbour_index[i], i, CR, F, gamma, upper_bound)
			child_fit[i] = fitness(data, child[i])

			#Actualizamos punto ideal
			ideal_point = [min(child_fit[i][j],ideal_point[j]) for j in range(2)]

    		#Actualizamos vecindario
			for j in range(n_neighbors):
				index = int(neighbour_index[i][j])
				new1 = weight[index,0] * abs(child_fit[i][0] - ideal_point[0])
				new2 = weight[index,1] * abs(child_fit[i][1] - ideal_point[1])
				new_te = max(new1, new2)
				old1 = weight[index,0] * abs(parent_fit[index][0]-ideal_point[0])
				old2 = weight[index,1] * abs(parent_fit[index][1]-ideal_point[1])
				old_te= max(old1, old2)

				if(new_te <= old_te):
					population.setIndiv(child[i], index)
					parent_fit[index][:]  = child_fit[i][:]

		n_eval+=1
	

	# Busco Best Compromise Solution
	max_fitness = [max(np.transpose(parent_fit)[0]), max(np.transpose(parent_fit)[1])]
	min_fitness = [min(np.transpose(parent_fit)[0]), min(np.transpose(parent_fit)[1])]

	fuzzy_matrix = np.zeros((population.pop_size,2))
	# Para cada miembro del conjunto óptimo de pareto
	for i in range(population.pop_size):
		# Para cada función objetivo
		for j in range(2):
			if parent_fit[i][j] <= min_fitness[j]:
				fuzzy_matrix[i][j] = 1
			elif parent_fit[i][j] >= max_fitness[j]:
				fuzzy_matrix[i][j] = 0
			else:
				fuzzy_matrix[i][j] = (max_fitness[j] - parent_fit[i][j]) / (max_fitness[j] - min_fitness[j])
		
	
	#Calculamos achievement degree para cada k
	total = np.matrix(fuzzy_matrix).sum()	
	achievement_deg = []
	for k in range(population.pop_size):			
		achievement_deg.append((fuzzy_matrix[k][0] + fuzzy_matrix[k][1]) / total)

	# Nos quedamos con el miembro con mayor achievement degree
	best_index = achievement_deg.index(max(achievement_deg))
	best_sol = population.population[best_index]

	# Represento gráficamente frontera de Pareto y mejor solución compromiso
	#plt.figure(figsize = (8, 6))
	#plt.plot(np.transpose(parent_fit)[0], np.transpose(parent_fit)[1],'r+')
	#plt.plot(parent_fit[best_index][0], parent_fit[best_index][1], 'bo')
	#plt.plot(ideal_point[0], ideal_point[1],'go')
	#plt.legend(('Soluciones óptimas de Pareto', 'Mejor solución compromiso','Punto ideal'), loc = 'upper right')
	#plt.xlabel('Distancia intraclase')
	#plt.ylabel('Distancia interclase')
	#plt.savefig("../output/pareto.png")

	return best_sol





def diffEvolution(data, population, neighbour_index, index, CR, F, gamma, upper_bound):
	"""
	Método que aplica Evolución Diferencial al individuo index de la población population
	"""
	child = population[index][:]

	# SELECCIONAMOS PADRES
	parents = getRandomParents(len(neighbour_index))
	indexes = [int(neighbour_index[parents[0]]), int(neighbour_index[parents[1]]), int(neighbour_index[parents[2]])]

	# Para cada gen del individuo
	for g in range(len(population[index])):

		# RECOMBINACIÓN DISCRETA
		n_random = random.random()
		if n_random < CR:
			# MUTACIÓN
			gen = mutate(g, F, population, indexes)
			# Comprobamos que no se sale del rango
			if gen > upper_bound:
				gen = upper_bound
			elif gen < gamma:
				gen = 0
			child[g] = gen	
		else:
			child[g] = population[index][g]


	return child





# Método que obtiene los índices de tres individuos aleatorios de la población
# Recibe el tamaño de la población
def getRandomParents(pop_size):

	v_index = [random.randint(0,pop_size-1), random.randint(0,pop_size-1)]

	# Si los índices obtenidos coinciden, obtenemos aleatorios hasta que sean diferentes
	while(v_index[0] == v_index[1]):
		v_index[1] = random.randint(0,pop_size-1)

	v_index.append(random.randint(0,pop_size-1))
	while(v_index[0] == v_index[2] or v_index[1] == v_index[2]):
		v_index[2] = random.randint(0,pop_size-1)


	return v_index



# Método que realiza la mutación diferencial  del tipo rand/1
def mutate(i_gen, F, population, parents):

	gen = (population[parents[0]][i_gen]
		+ F*(population[parents[1]][i_gen] - population[parents[2]][i_gen]))

	return gen

