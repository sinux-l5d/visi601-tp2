import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation


class Grid:

    def __init__(self, nbrow, nbcol):
        """
        Constructeur. Fabrique une grille avec `nbrow` lignes et `nbcol` colonnes.
        """
        self.nbcol = nbcol
        self.nbrow = nbrow
        self.number()

    def number(self):
        """
        Numerote les pixels dans `self.index`, et pour un index `idx` mémorise
        sa ligne `self.I[ idx ]` et sa colonne `self.J[ idx ]`
        """
        self.I = [i for i in range(self.nbrow) for j in range(self.nbcol)]
        self.J = [j for i in range(self.nbrow) for j in range(self.nbcol)]
        # On crée la liste d'indice qui va de 0 à la taille de la matrice (cad nbrow*nbcol)

        # self.index = {}
        # for i in range(self.nbcol*self.nbrow):
        #     self.index[(self.J[i], self.I[i])] = i

        self.index = {(self.I[i], self.J[i]): i for i in range(self.nbcol*self.nbrow)}

    def getIndex(self, i, j):
        """
        Retourne l'indice du pixel `(i,j)`.
        """
        return self.index.get((i, j), -1)

    def getRow(self, idx):
        """
        Retourne la ligne du pixel ayant pour indice `idx`
        """
        return self.I[idx]

    def getCol(self, idx):
        """
        Retourne la colonne du pixel ayant pour indice `idx`
        """
        return self.J[idx]

    def _neighbors_of(self, i, j):
        yield i-1, j
        yield i+1, j
        yield i, j-1
        yield i, j+1

    def neighbors(self, idx):
        """
        Retourne la liste des voisins direct du sommet d'indice idx
        """
        N = []
        try:
            i = self.getRow(idx)
            j = self.getCol(idx)
        except IndexError:
            return []

        for ni, nj in self._neighbors_of(i, j):
            if (ni, nj) in self.index:
                N.append(self.getIndex(ni, nj))

        return N

    def size(self):
        """
        Taille n du vecteur u.
        """
        return len(self.I)

    def Identity(self):
        """
        Retourne la matrice identite de taille n*n
        """
        n = self.size()
        LIGS = []  # les lignes des coefficients
        COLS = []  # les colonnes des coefficients
        VALS = []  # les valeurs des coefficients
        for idx in self.index.values():
            LIGS.append(idx)
            COLS.append(idx)
            VALS.append(1.0)
        # print(LIGS, COLS, VALS )
        M = coo_matrix((VALS, (LIGS, COLS)), shape=(n, n))
        return M.tocsc()

    def LaplacianD(self):
        """ Retourne le laplacien de Dirichlet et retourne la matrice creuse 
            --> juste changer les -(len.. ) en -4 constant
        """
        ...

    def Laplacian(self):
        """ Retourne le laplacien et retourne la matrice creuse """
        n = self.size()
        LIGS = []  # les lignes des coefficients
        COLS = []  # les colonnes des coefficients
        VALS = []  # les valeurs des coefficients

        # on parcourt les indices
        for k in self.index.values():
            # on calcule les voisins de l'indice
            voisins = self.neighbors(k)
            nbvoisins = len(voisins)

            # on met pour valeur - le nombre des voisins si on est sur (i,i)
            LIGS.append(k)
            COLS.append(k)
            VALS.append(-len(voisins))

            # on met pour valeur 1 si on est sur une colonne d'un voisin
            for v in voisins:
                LIGS.append(k)
                COLS.append(v)
                VALS.append(1.0)

        L = coo_matrix((VALS, (LIGS, COLS)), shape=(n, n))
        return L.tocsc()

    def explicitEuler(self, U0, T, dt):
        """ A partir du vecteur de valeurs U0, calcule U(T) en itérant des pas dt successifs.  """
        Id = self.Identity()
        U = U0
        L = self.Laplacian()

        for _ in np.arange(0, T, dt):
            U = (Id + dt*L) * U

        return U

    def vectorToImage(self, V):
        img = np.zeros((self.nbrow, self.nbcol))
        K = self.index.keys()
        I = self.index.values()
        for k, idx in zip(K, I):
            img[k[0], k[1]] = V[idx]
        return img

    def implicitEuler(self, U0, T, dt):
        """"
        A partir du vecteur de valeurs U0, calcule U(T) en itérant des pas dt successifs. 
        permet d'obtenir une marge d'erreur plus petite par rapport a explicitEuler  
        """
        Id = self.Identity()
        U = np.array(U0)
        L = self.Laplacian()
        lu = splu(Id - dt * L)

        for _ in np.arange(0, T, dt):
            U = lu.solve(U)

        return U

    def imageToVector(self, img):
        """"methode qui permet de transformer une image en noir et blanc en un vecteur"""
        V = []
        for i in range(self.nbrow):
            V.extend(img[i, :])
        return V

    def diffuseImage(self, Img, T, dt):
        """
        A partir d'une image N&B Img, diffuse cette image pendant un temps T, par pas dt, et retourne l'image résultante
        """
        vecteur = self.imageToVector(Img)
        diffusion = self.implicitEuler(vecteur, T, dt)
        return self.vectorToImage(diffusion)

    def diffuseImageRGB(self, Img, T, dt):
        """
        A partir d'une image N&B Img, diffuse cette image pendant un temps T, par pas dt, et retourne l'image résultante
        """
        red = self.diffuseImage(Img[:, :, 0], T, dt)
        green = self.diffuseImage(Img[:, :, 1], T, dt)
        blue = self.diffuseImage(Img[:, :, 2], T, dt)
        return np.stack((red, green, blue), axis=2)


def testlaplacian():
    D = Grid(4, 4)
    print(f'{D.I=}')
    print(f'{D.J=}')
    print(f'{D.index=}')
    print(f'{D.neighbors(2)=}')
    l = D.Laplacian()
    print('l=')
    print(l)


def affiche_animation():
    D = Grid(10, 10)
    V = [0.] * D.size()
    V[42] = 5.0
    V[37] = 10.0

    fig = plt.figure()
    ims = []

    for _ in range(50):
        img = D.vectorToImage(V)
        im = plt.imshow(img, cmap='hot', vmin=0.0, vmax=1.0, animated=True)
        ims.append([im])
        V = D.implicitEuler(V, 0.1, 0.1)

    ani = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat_delay=1000)
    # ani.save("movie.gif")
    plt.show()


def diffusionimg():
    # pour charger une image dans un tableau numpy.array
    img = mpimg.imread('images/mandrill-240-b02.png')
    largeur = img.shape[0]
    hauteur = img.shape[1]
    D = Grid(largeur, hauteur)
    img_diffuse = D.diffuseImageRGB(img, 20.0, 5.0)
    # im = plt.imshow(img_diffuse, cmap='hot', vmin=0.0, vmax=1.0)
    plt.imshow(img_diffuse)
    plt.show()


if __name__ == "__main__":
    diffusionimg()
    # print(D.Identity())
    # print(D.Laplacian())