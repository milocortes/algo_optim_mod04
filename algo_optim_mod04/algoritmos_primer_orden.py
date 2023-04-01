import numpy as np 

def steepest_descent(gradiente, punto_inicial : np.array, rho : float, maxiter : int) -> np.array:
    """
    Funcion : steepest_descent

    Argumentos : 
        * gradiente : Gradiente de la función a minimizar
        * punto_inicial :  Punto inicial para iniciar la búsqueda (np.array)
       * rho :  Tasa de aprendizaje (float)
       * maxiter : Número de iteraciones (int) 
    """

    X = punto_inicial

    for it in range(maxiter):
        if it % 10 == 0:
          print(f"Iteración {it}")

        # Utilizamos la regla de actualización
        X = X - rho*gradiente(*X)

    return X