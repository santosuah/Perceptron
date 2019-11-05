import numpy as np
from perceptron import Perceptron
from banner import generar_banner

# Entradas
p1 = "2.7810836 1.465489372 3.396561688 1.38807019 3.06407232 7.627531214 5.332441248 6.922596716 8.675418651 7.673756466"
p2 = "2.550537003 2.362125076 4.400293529 1.850220317 3.005305973 2.759262235 2.088626775 1.77106367 -0.242068655 3.508563011"
p = np.matrix(p1 + ";" + p2)

# Objetivos
t1 = "0 0 0 0 0 1 1 1 1 1"
t = np.matrix(t1)

print("\n" + generar_banner("Vectores(p) y objetivos(t)") + "\n")
for i in range(p.shape[1]):
    print(" ", p[:,i].tolist(), "=>", t[:,i])

# Perceptron
perceptron = Perceptron(p, t)

# Entrenar
print("\n" + generar_banner("Entrenamiento del perceptrón") + "\n")
perceptron.entrenar(lr=0.2, debug=True)

# Clasificar vectores mediante perceptron entrenado
print("\n" + generar_banner("Clasificar con perceptrón") + "\n")
for i in range(p.shape[1]):
    y = perceptron.simular(p[:,i])
    print(" ", p[:,i].tolist(), "=>", y)
print()

# Representar gráficamente
perceptron.representar()
