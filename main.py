# -------------------------------------------------------------------------------
# Name:       TP 2 Génie mathématique Ma223
#
# Etudiant 1 : Kylian Forsans
# Etudiant 2 : Tony Raliterason
#
# Classe : 2PG
# Groupe : G1
#
# -------------------------------------------------------------------------------

from graphe_temps import *
from graphe_erreur import *

A = np.array([[4, -2, -4], [-2, 10, 5], [-4, 5, 6]])
B = np.array([[6], [-9], [-7]])

print("Cholesky :\n", ResolCholesky(A, B))
print("Cholesky machine :\n", Cholesky_machine(A, B))
print("Gauss :\n", Gauss(A, B))
print("Gauss machine :\n", np.transpose(Gauss_machine(A, B)))
print("LU :\n", ResolutionLU(A, B))

Courbe_de_temps()
Courbe_d_erreur()
