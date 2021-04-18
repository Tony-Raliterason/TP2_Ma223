import numpy as np
from math import sqrt

def symetrique(n):
    """Permet d'obtenir une matrice symétrique définie positive"""
    global A
    n = int(n)

    pas_trouve = True

    while pas_trouve:
        M = np.random.rand(n, n)
        Mt = np.transpose(M)
        if np.linalg.det(M) != 0:
            A = np.dot(M, Mt)
            pas_trouve = False

    return A

def Cholesky(A):
    """Permet de décomposer la matrice A selon la méthode de Choleski"""
    n, m = A.shape

    # Crée une matrice nulle pour L
    L = np.zeros((n, m), dtype=float)

    # Méthode de décomposition de Cholesky
    for i in range(n):  # i = ligne
        for k in range(i+1):    # k = colonne
            somme = 0
            for j in range(k):  # pour une somme allant de j=1 à k-1
                somme += L[i, j] * L[k, j]
            if i == k: # Eléments diagonaux
                L[i, k] = sqrt(A[k, k] - somme)
            else:
                L[i, k] = (1.0 / L[k, k] * (A[i, k] - somme))
    return L


def ResolutionSystTriInf(L, B):
    Maug = np.column_stack([L, B])

    n, m = Maug.shape

    if m != n + 1:
        print("Il ne s'agit pas d'une matrice augmentée")
        return

    Y = np.zeros(n)
    for lp in range(n):
        somme = 0
        for i in range(lp):
            somme += Y[i] * Maug[lp, i]
        Y[lp] = (Maug[lp, -1] - somme) / (Maug[lp, lp])

    return Y

def ResolutionSystTriSup(Lt, Y):
    Maug = np.column_stack([Lt, Y])

    n, m = Maug.shape

    if m != n + 1:
        print("Il ne s'agit pas d'une matrice augmentée")

    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        somme = 0
        for k in range(i, n):
            somme += X[k] * Maug[i, k]
        X[i] = (Maug[i, -1] - somme) / Maug[i, i]

    return X


def ResolCholesky(A, B):
    """Permet de résoudre un système avec la méthode de Choleski"""

    L = Cholesky(A)
    Lt = np.transpose(L)
    Y = ResolutionSystTriInf(L, B)
    print("Y = ", Y)
    X = ResolutionSystTriSup(Lt, Y)

    return X

##################################################################

def Cholesky_machine(A, B):
    L = np.linalg.cholesky(A)
    Lt = np.transpose(L)
    Y = ResolutionSystTriInf(L, B)
    X = ResolutionSystTriSup(Lt, Y)

    return X

#################################################################

def ReductionGauss(Aaug):
    if len(Aaug) == len(Aaug[0]) - 1:
        n = len(Aaug)
        for i in range(0, n-1):
            for k in range(i+1, n):
                pivot = Aaug[k, i] / Aaug[i, i]
                for j in range(0, n+1):
                    Aaug[k, j] = Aaug[k, j] - pivot * Aaug[i, j]
        return Aaug
    else:
        print("Réduction impossible, il ne s'agit pas d'une matrice augmentée")

def ResolutionSystTriSupGauss(Taug):
    n, m = np.shape(Taug)

    if m != n + 1:
        print("Il ne s'agit pas d'une matrice augmentée")
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        somme = 0
        for k in range(i+1, n):
            somme += x[k] * Taug[i, k]
        x[i] = (Taug[i, -1] - somme) / Taug[i, i]

    return x

def Gauss(A, B):
    Aaug = np.column_stack([A, B])
    Taug = ReductionGauss(Aaug)
    x = ResolutionSystTriSupGauss(Taug)
    return x

#################################################################

def Gauss_machine(A, B):
    X = np.linalg.solve(A, B)
    return X

#################################################################

def DecompositionLU(A):
    n, m = A.shape
    L = np.eye(n)
    U = np.copy(A)

    for i in range(0, n-1):
        for k in range(i+1, n):
            pivot = U[k, i] / U[i, i]
            L[k, i] = pivot
            for j in range(i, n):
                U[k, j] = U[k, j] - pivot * U[i, j]
    return L, U

def ResolutionLU(A, B):
    L = DecompositionLU(A)[0]
    U = DecompositionLU(A)[1]
    Y = ResolutionSystTriInf(L, B)
    X = ResolutionSystTriSup(U, Y)

    return X

#################################################################

