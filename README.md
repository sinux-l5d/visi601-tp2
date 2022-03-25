Juliette NEYRAT & Simon LEONARD (CMI INFO L3)


# VISI601 | TP2 | Construction de l’opérateur Laplacien

## Une classe Grid pour représenter le domaine

#### La méthode Grid.number()
Elle permet de numeroter les pixels dans self.index.
Pour cela, nous avons crée la liste d'indice qui va de 0 à la taille de la matrice (c'est a dire nbrow*nbcol)
Il s'agit donc d'un dictionnaire possédant les coordonnées en clé et l'indice en valeur.

## Voisins dans la grille
#### La méthode Grid.neighbors(indice):
Cette fonction a pour but de nous retourner la liste des voisins directs du sommet d'indice donné en paramètre.
Pour cela, on recupere la ligne et la colonne de l'indice donné.
On parcourt les coordonnées (i-1,j), (i+1, j), (i, j-1) et (i, j+1) a l'aide d'une fonction _neighbors_coords(self, i, j)
Et ainsi, si les coordonnées retournées existent bien dans self.index, alors on les ajoute a notre liste de retour.

## Construction de l’opérateur Laplacien
#### La méthode Grid.LaplacianD():
Retourne le laplacien de Dirichlet et retourne la matrice creuse 
On a plusieurs cas : 
    - si on est sur la diagonale (i,i): 
        on met la valeur -4.0.
    - si on est sur une colonne d'un voisin, alors : 
        on met la valeur 1.0.

#### La méthode Grid.Laplacian():
Il s'agit de la meme fonction, simplement lorsque l'on est sur la diagonale, on retourne - (le nombre de voisin- de la coordonnée et non pas -4.


# Processus de diffusion
## Diffusion simple par Euler explicite
#### La méthode Grid.explicitEuler(u0,T,dt):
On calcule simplement U = (Id + dt*L) * U avec :
    U = U0 à l'initialisation
    Id : la matrice identite 
    dt : donnee 
    L : le Laplacian du Grid
pour un temps T et un intervalle dt.
On remarque que lorsqu'on augmente le pas de dt alors, la diffusion s'effectue plus rapidement

## Diffusion simple par Euler implicite
#### La méthode Grid.implicitEuler(u0,T,dt):
A partir du vecteur de valeurs U0, calcule U(T) en itérant des pas dt successifs. 
Cela permet d'obtenir une marge d'erreur plus petite par rapport a explicitEuler avec l'utilistion du Laplacian.

## Diffusion sur une image
#### La méthode Grid.diffuseImage(Img,T,dt):
 A partir d'une image N&B Img, diffuse cette image pendant un temps T, par pas dt, et retourne l'image résultante
Pour cela, on cree un vecteur a l'aide de imageToVector(img).
Puis on diffuse a l'aide d'implicitEuler. On retourne donc le vetceur cree de la diffusion. 
#### La méthode Grid.diffuseImageRGB(Img,T,dt):
Nous reutilisons simplement diffuseImage() sur chacunes de nos couleurs.
C'est a dire pour le vecteur vert, rouge et bleu
Puis a partir de cela, on cree une matrice avec les trois vecteurs, a l'aide de stack.

## Diffusion comme minimisation d’une fonctionnelle
#### La méthode EQM( u, g):
On applique simplement la formule donnée c'est a dire : 
1/(3*u.shape[0]) * np.sum((vecU-vecG)**2) avec :
    : vecU l'image a partirt du vecteur U 
    : vecG l'image a partirt du vecteur G

