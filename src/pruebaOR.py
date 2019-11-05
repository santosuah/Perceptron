import numpy as np
from perceptron import Perceptron
from banner import generar_banner

# Función OR
# Entradas
p1 = "0 0 1 1"
q1 = "0 1 0 1"
p = np.matrix(p1 + ";" + q1)

# Objetivos
t1 = "0 1 1 1"
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
