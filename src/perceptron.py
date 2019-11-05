import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):

	def __init__(self, entradas, objetivos):

		self.p = entradas
		self.t = objetivos

		self.numero_neuronas = 1

		self.numero_entradas = self.p.shape[0]
		self.numero_salidas = self.t.shape[0]

		# incicializar con las dimensiones (SxR)
		self.w = np.random.rand(self.numero_neuronas, self.numero_entradas).T

		# inicializar la bía con dimensiones (Sx1)
		self.b = np.random.rand(self.numero_neuronas, 1)

		self.tab = "   "


	# evaluar una entrada en la red
	def simular(self, entrada):
		return self._hardlim(self.w.T * entrada + self.b)


	# función escalón
	def _hardlim(self, n):
		if n > 0:  r = 1
		else:      r = 0
		return r


	def entrenar(self, lr, max_epoch=1000, debug=False):

		# época actual
		epoca = 1

		# instante de tiempo
		k = 0

		# número de vectores correctamente clasificados consecutivos
		aciertos = 0

		# condición de parada de entrenamiento
		convergencia = False

		# función hardlim vectorizada
		vhardlim = np.vectorize(self._hardlim)


		if debug:
			print(" - Valores iniciales")
			print(self.tab + "w(" + str(k) + ") =", self.w.tolist())
			print(self.tab + "b(" + str(k) + ") =", self.b.tolist())

		# mientras que la red no converja
		# (clasifique correctamente los vectores)
		# o no llegue al máximo número de epocas indicado
		while (not convergencia) and (epoca <= max_epoch):

			if debug: print("\n- Época", epoca)

			# recorremos los ejemplos de entrenamiento
			for i in range(self.p.shape[1]):

				if debug: 
					print("\n" + self.tab + "+ Iteración", i+1)
					kstr = str(k)
					k1str = str(k+1)

				# calcular la salida de la red para el vector j
				a = vhardlim(self.w.T * self.p[:,i] + self.b)
				if debug: print(2*self.tab + "a(" + kstr + ") = hardlim(w(" + kstr + ") * p" + kstr + " + b(" + kstr + ")) =", a)

				# calcular error entre el objetivo y el resultado
				e = self.t[:,i] - a
				if debug: print(2*self.tab + "e(" + kstr + ") = (t(" + kstr + ") - a(" + kstr + ")) =", e)

				# ajustar pesos si existe error
				if e != 0:
					if debug: print("\n" + 2*self.tab + "Actualizar w y b")

					# reseteo el número de aciertos
					aciertos = 0

					# ajuste w y b
					self.w = self.w + lr * e.item() * self.p[:,i]
					self.b = self.b + lr * e.item()

					if debug:
						print(2*self.tab + "w(" + k1str + ") = w(" + kstr + ") + lr * e(" + kstr + ") * p" + kstr + " =", self.w.tolist())
						print(2*self.tab + "b(" + k1str + ") = b(" + kstr + ") + lr * e(" + kstr + ") =", self.b.tolist())

				else:
					aciertos += 1

					if debug:
						print(2*self.tab + "w(" + k1str + ") = w(" + kstr + ")")
						print(2*self.tab + "b(" + k1str + ") = b(" + kstr + ")")

				# si todos los vectores de entrada son correctamente
				# clasificados, la red está entrenada
				if aciertos == self.t.shape[1]:
					convergencia = True
					break

				k += 1

			epoca += 1


	def representar(self):

		# Si los vectores de entrada son de dos dimensiones
		if self.p.shape[0] == 2:

			p1 = np.linspace(self.p[0].min() - 0.1, self.p[0].max() + 0.1, 20)
			# p2 = (-(self.w[0].item() * p1) - self.b.item()) / self.w[1].item()
			p2 = -(self.w[0].item() / self.w[1].item()) * p1 - (self.b.item() / self.w[1].item())

			for i in range(self.p.shape[1]):
				punto = self.p[:,i]
				target = self.t[:,i]

				if target: c = "red"
				else: c = "blue"

				plt.scatter(punto.flat[0], punto.flat[1], color=c)


			axes = plt.gca()
			axes.set_xlim([self.p[0,:].min() - 0.1, self.p[0,:].max() + 0.1])
			axes.set_ylim([self.p[1,:].min() - 0.1, self.p[1,:].max() + 0.1])

			plt.plot(p1, p2, "black")

			plt.xlabel("p1", color="#1C2833")
			plt.ylabel("p2", color="#1C2833")

			plt.grid()
			plt.show()

		else:
			print("Borde de decisión no representable")
