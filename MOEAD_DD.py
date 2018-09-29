#######################################################
# Autora: Elena Romero Contreras
# Implementación del algoritmo MOEAD con funciones objetivo
# distancia intra e interclase
#######################################################

import sys
import random
import data as dt
import population as pop
import fitness_DD as fit
import numpy as np
import time
import MOEAD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split


def MOEAD_DD(data=None, train=None, test=None, POP_SIZE=100, CR=0.95, F=0.7, MAX_EVAL=150, gamma=1, upper_bound=10):
	"""
	Algoritmo MOEAD con funciones objetivo distancia intraclase e interclase
	"""
	if data:
		X_train, X_test, y_train, y_test = train_test_split(data.values, data.label, test_size=0.2, random_state=42)
		train = dt.Data(X= X_train, y= y_train)
		test = dt.Data(X= X_test, y= y_test)

	start = time.time()
	train.normalize(upper_bound)
	test.normalize(upper_bound)

	# Calculamos distancias intraclase e interclase
	train.calculateDistIntra()
	train.calculateDistInter()

	# MOEA/D
	w = MOEAD.MOEAD(data=train, pop_size=POP_SIZE, fitness=fit.fitness,
		CR=CR, F=F, MAX_EVAL=MAX_EVAL, gamma=gamma, upper_bound = upper_bound)
	
	#Seleccionamos y ponderamos características
	train.weight(w)
	test.weight(w)

	#----------- TEST -----------
	
	## 5-NN ##
	knn = KNeighborsClassifier(n_neighbors=5) 
	knn.fit(train.values, train.label)
	pred_label = knn.predict(test.values)
	end = time.time()

	# Medición resultados
	n_success = sum(np.array(pred_label == test.label))

	accuracy = n_success/test.n_row
	n_features = sum(np.array(w != np.zeros(len(w))))
	t =end-start

	return (accuracy, n_features, t)



if __name__ =='__main__':

	fname = sys.argv[1]

	if len(sys.argv)>2:
		label_first = sys.argv[2]
	else:
		label_first = None
		
	random.seed(123)
	
	# Lectura de datos
	X = dt.Data(fname=fname, label_first = label_first)
	
	# 5-fold cross validation
	kf = KFold(n_splits = 5, shuffle=True, random_state= 1234)

	#Inicializamos medidas
	total_accuracy = []
	total_n_features = []
	total_time = []

	for train_index, test_index in kf.split(X.values):

		train = dt.Data(X= X.values[train_index], y= X.label[train_index])
		test = dt.Data(X= X.values[test_index], y= X.label[test_index])

		accuracy, n_features, t = MOEAD_DD(train = train, test = test)

		total_accuracy.append(accuracy)
		total_n_features.append(n_features)
		total_time.append(t)

	print(fname, "\t{0:.5f}".format(np.mean(total_accuracy)), 
		"\t", np.mean(total_n_features), "\t{0:.5f}".format(np.mean(total_time)))
