from cholesky import *
import time as time
import matplotlib.pyplot as plt


def Courbe_de_temps():
    indices = []
    y = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []

    for i in range(100, 400, 50):
        A = symetrique(i)
        B = np.random.rand(i, 1)

        # Cholesky
        temps_initial = time.time()
        ResolCholesky(A, B)
        temps_final = time.time()
        temps = temps_final - temps_initial

        # Gauss
        temps_initial2 = time.time()
        Gauss(A, B)
        temps_final2 = time.time()
        temps2 = temps_final2 - temps_initial2

        # Décomposition LU
        temps_initial3 = time.time()
        ResolutionLU(A, B)
        temps_final3 = time.time()
        temps3 = temps_final3 - temps_initial3

        # Cholesky machine
        temps_initial4 = time.time()
        Cholesky_machine(A, B)
        temps_final4 = time.time()
        temps4 = temps_final4 - temps_initial4

        # Gauss machine
        temps_initial5 = time.time()
        Gauss_machine(A, B)
        temps_final5 = time.time()
        temps5 = temps_final5 - temps_initial5

        # Ajout des courbes
        indices.append(i)
        y.append(temps)
        y2.append(temps2)
        y3.append(temps3)
        y4.append(temps4)
        y5.append(temps5)

    plt.plot(indices, y, color='r', label="Cholesky")
    plt.plot(indices, y2, color='b', label="Gauss")
    plt.plot(indices, y3, color='y', label="LU")
    plt.plot(indices, y4, color='green', label="Cholesky machine")
    plt.plot(indices, y5, color='brown', label="Gauss machine")

    plt.title("Temps de calcul de la matrice en fonction de sa dimension")

    plt.xlabel("Dimension")
    plt.ylabel("Temps (sec)")

    plt.legend()

    plt.show()
