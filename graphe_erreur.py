from cholesky import *
import matplotlib.pyplot as plt


def Courbe_d_erreur():
    indices = []
    y = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []

    for i in range(100, 400, 50):
        A = symetrique(i)
        B = np.random.rand(i, 1)
        C = np.copy(A)

        ResolCholesky(A, B)
        erreur = np.linalg.norm(A @ ResolCholesky(A, B) - np.ravel(B))

        Gauss(A, B)
        erreur2 = np.linalg.norm(A @ Gauss(A, B) - np.ravel(B))

        ResolutionLU(A, B)
        erreur3 = np.linalg.norm(C @ ResolutionLU(A, B) - np.ravel(B))

        Cholesky_machine(A, B)
        erreur4 = np.linalg.norm(A @ Cholesky_machine(A, B) - np.ravel(B))

        np.linalg.solve(A, B)
        erreur5 = np.linalg.norm(A @ np.linalg.solve(A, B) - np.ravel(B))

        indices.append(i)
        y.append(erreur)
        y2.append(erreur2)
        y3.append(erreur3)
        y4.append(erreur4)
        y5.append(erreur5)

    plt.plot(indices, y, color='r', label="Cholesky")
    plt.plot(indices, y2, color='b', label="Gauss")
    plt.plot(indices, y3, color='y', label="LU")
    plt.plot(indices, y4, color='green', label="Cholesky machine")
    plt.plot(indices, y5, color='brown', label="Gauss machine")

    plt.title("         Erreur de calcul en fonction de la dimension de la matrice")

    plt.xlabel("Dimension de la matrice")
    plt.ylabel("Erreur de calcul")

    plt.legend()

    plt.show()
