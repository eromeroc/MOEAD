#######################################################
# Autora: Elena Romero Contreras
# Clase Data
#######################################################


import numpy as np
from numpy import array, shape, where, in1d
from math import log
from sklearn import preprocessing

class Data:
	"""
	Clase que representa un conjunto de datos donde cada fila es una instancia y
	cada columna una característica.
	Tiene métodos para leer, normalizar y calcular algunas propiedades de los datos
	"""


	def __init__(self, fname=None, X=None, y=None, label_first=None):

		"""
		Constructor de la clase Data
		Lee los datos de un fichero dado
		o crea conjunto de datos a partir de valores y etiquetas dadas
		"""
		if fname:
			self.values = []		# Matriz de instancias
			self.label = []			# Vector de etiquetas

			file = open(fname, 'r').read().splitlines()

			for line in file:
				current_line=line.split(",")
				
				if label_first:	#Si la etiqueta está al principio de línea
					self.label.append(current_line.pop(0))
				else:
					self.label.append(current_line.pop())

				v_data = []
				for i in current_line:
					if i=='?':
						v_data.append(0)
					else:
						v_data.append(float(i))

				self.values.append(np.array(v_data))

			self.values = np.array(self.values)
			self.label = np.array(self.label)

		else:
			self.values = X
			self.label = y

		self.n_row = len(self.values)		# Nº de instancias
		self.n_col = len(self.values[0])	# Nº de atributos


	def encodeLabel(self):
		"""
		Codifica etiquetas
		"""
		le = preprocessing.LabelEncoder()
		le.fit(self.label)
		self.label = np.array(le.transform(self.label))


	def normalize(self, upper_bound):
		"""
		Método que normaliza el conjunto de datos
		"""
		min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,upper_bound))
		self.values = min_max_scaler.fit_transform(self.values)



	def calculateDistIntra(self):
		"""
		Método que calcula distancia intraclase total entre las instancias de un conjunto de datos
		"""
		dist = 0

		for i in range(self.n_row):
			for j in range(i+1, self.n_row): #Matriz simétrica->recorremos solo mitad

				if self.label[i] == self.label[j]: #Si son de la misma clase
					dist += abs(self.values[i]- self.values[j])

		self.dist_intra = dist



	def calculateDistInter(self):
		"""
		Método que calcula distancia interclase total entre las instancias de un conjunto de datos
		"""
		dist = 0

		for i in range(self.n_row):
			for j in range(i+1, self.n_row): 

				if self.label[i] != self.label[j]: #Si son de distinta clase
					dist += abs(self.values[i]- self.values[j])

		self.dist_inter = dist


	def weight(self, w):
		"""
		Método que selecciona y pondera las características según vector w dado
		"""
		for i in range(self.n_row):
			self.values[i] = self.values[i] * w



	def calculateP(self,x):
		"""
		Calcula la probabilidad de que aparezca cada uno de los elementos del vector x
		"""
		values = sorted(set(x))
		p = [shape(where(x==value))[1] /len(x) for value in values]

		return p


	def calculatePY(self): 
		"""
		Calcula la probabilidad de que aparezca cada elemento del vector de etiquetas
		"""		
		self.py = self.calculateP(self.label)



	def calculatePX(self):
		"""
		Calcula la probabilidad de que aparezca cada elemento de cada atributo
		"""
		values = self.values.transpose() #Cada fila es un atributo
		self.px = [self.calculateP(i) for i in values] #Calcula vector de probabilidades



	def calculateRed(self):
		"""
		Calcula la redundancia entre cada par de atributos
		"""
		values = self.values.transpose()
		self.redundancy = np.zeros((self.n_col, self.n_col))

		for i in range(self.n_col-1):
			for j in range(i+1, self.n_col):
				self.redundancy[i][j] = self.mutual_information(values[i], values[j], self.px[i], self.px[j])




	def calculateRel(self):
		"""
		Calcula la relevancia entre cada atributo y las etiquetas
		"""
		values = self.values.transpose()
		self.relevancy = []

		for i in range(self.n_col):
			self.relevancy.append(self.mutual_information(values[i], self.label, self.px[i], self.py))



	def mutual_information(self, x, y, px, py):
		"""
		Calcula la información mutua entre dos vectores
		"""
		summation = 0.0

		values_x = list(sorted(set(x)))
		values_y = list(sorted(set(y)))

		for i in range(len(values_x)):
			for j in range(len(values_y)):
				pxy = len(where(in1d(where(x==values_x[i])[0], where(y==values_y[j])[0])==True)[0]) / len(x)

				if pxy > 0.0:
					summation += pxy * log((pxy/ (px[i]*py[j])), 10)

		return summation

		





