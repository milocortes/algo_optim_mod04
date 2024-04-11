from typing import Callable
import numpy as np 


class PrimerOrden:
    """
    La clase PrimerOrden es la clase padre de los optimizadores de primer orden.
    
        La clase incorpora los elementos imprescindibles de cualquier método de 
        optimización de primer orden : 
            * Valor inicial, 
            * Ciclo iterativo,
            * Tasa de aprendizaje,
            * Función a minimizar,
            * Derivada de la función a derivar

        Estos elementos son heredados a las clases hijo que, para el caso de este paquete, 
        corresponden a los siguientes métodos de optimización de primer orden:
            * Steepest descent
            * Momentum
            * Momentum Nesterov
            * Adam

    Argumentos de inicializacion
    _________________________________________________________________________________

        - f         :   Función a minimizar
        - df        :   Derivada de la función a minimizar
        - rho       :   Tasa de aprendizaje
        - x         :   Valor inicial de la búsqueda
        - maxiter   :   Cantidad máxima de iteraciones
        - args      :   Tupla que incorpora argumentos adicionales para el gradiente

    """

    def __init__(self, f : Callable, df : Callable, rho : float, x : np.array, maxiter : int , args = ()):
        self.f = f
        self.df = df 
        self.rho = rho 
        self.x = x 
        self.maxiter = maxiter
        self.x0 = x
        self.x_records = []
        self.args = args

    def itera(self):
        """
        El método itera() ejecuta el algoritmo iterativo la cantidad de iteraciones máximas
        definida por el usuario.
        """

        ## Iteramos el algoritmo 
        for it in range(self.maxiter):
            ## El método actualiza() utiliza la regla de actualización particular a cada método de optimización         
            self.actualiza(it)

    def optimiza(self) -> np.array:
        """    
        El método optimiza() ejecuta la iteración del algoritmo y regresa el valor
            arreglo de x que minimiza la función
        """

        ## Reiniciamos los valores iniciales
        self.x = self.x0
        self.x_records = []

        self.itera()

        return self.x

class Steepest(PrimerOrden):
    """
    La clase Steepest hereda la implementación de la clase padre PrimerOrden. 
        La clase Steepest implementa la regla de actualización del método Steepest Descent

    NOTA: La clase no necesita argumentos de inicialización dado que toda la información 
        necesitada por el método está en los argumentos de inicialización de la clase padre
        
    """

    def actualiza(self, it : int):
        """
        El método actualiza() implementa la regla de actualización particular al método Steepest Descent
        """
        self.x_records.append(self.x)
        self.x = self.x - self.rho*self.df(self.x, *self.args)

class Momentum(PrimerOrden):
    """
    La clase Momentum hereda la implementación de la clase padre PrimerOrden. 
        La clase SteepMomentumest implementa la regla de actualización del método de Momentum
        
    Argumentos de inicializacion
    _________________________________________________________________________________

        - m         :   Término de momentum
        - beta      :   Factor de escala o de intensidad de inercia
    
    """

    def __init__(self, m : np.array, beta : float, f : Callable, df : Callable, rho : float, x : np.array, maxiter : int, args = ()):
        super().__init__(f, df, rho, x, maxiter, args)
        self.m = m 
        self.beta = beta
        

    def actualiza(self, it : int):
        """
        El método actualiza() implementa la regla de actualización particular al método de Momentum
        """
        self.x_records.append(self.x)

        self.m = self.beta*self.m + self.df(self.x, *self.args)
        self.x = self.x - self.rho*self.m

class MomentumNesterov(PrimerOrden):
    """
    La clase MomentumNesterov hereda la implementación de la clase padre PrimerOrden. 
        La clase MomentumNesterov implementa la regla de actualización del método de Momentum Nesterov
        
    Argumentos de inicializacion
    _________________________________________________________________________________

        - m         :   Término de momentum
        - beta      :   Factor de escala o de intensidad de inercia
    
    """

    def __init__(self, m : np.array, beta : float, f : Callable, df : Callable, rho : float, x : np.array, maxiter : int, args = ()):
        super().__init__(f, df, rho, x, maxiter, args)
        self.m = m 
        self.beta = beta
        

    def actualiza(self, it : int):
        """
        El método actualiza() implementa la regla de actualización particular al método de Momentum
        """
        
        self.x_records.append(self.x)

        self.m = self.beta*self.m - self.rho*self.df(self.x + self.beta*self.m, *self.args)
        self.x = self.x + self.m

class Adam(PrimerOrden):
    """
    La clase Adam hereda la implementación de la clase padre PrimerOrden. 
        La clase Adam implementa la regla de actualización del método de Adam.
        
    Argumentos de inicializacion
    _________________________________________________________________________________

        - m             :   Término de momentum del gradiente
        - s             :   Término de momentum del gradiente al cuadrado
        - beta_uno      :   Factor de ponderación para el cálculo del promedio móvil del gradiente
        - beta_dos      :   Factor de ponderación para el cálculo del promedio móvil del cuadrado del gradiente
        - epsilon       :   Término agregado al denominador para mejorar la estabilidad numérica
    
    """

    
    def __init__(self, m : np.array, s : np.array, beta_uno : float, beta_dos : float, epsilon : float, f : Callable, df : Callable, rho : float, x : np.array, maxiter : int, args = ()):
        super().__init__(f, df, rho, x, maxiter, args)
        self.m = m 
        self.s = s
        self.beta_uno = beta_uno
        self.beta_dos = beta_dos 
        self.epsilon  = epsilon
        self.m_tilde = None
        self.s_tilde = None
    
    def actualiza(self, it : int):
        """
        El método actualiza() implementa la regla de actualización particular al método de Momentum
        """
        self.x_records.append(self.x)

        self.m = self.beta_uno*self.m + self.beta_uno*(1-self.beta_uno)*self.df(self.x, *self.args)
        self.s = self.beta_dos*self.s + self.beta_dos*(1-self.beta_dos)*self.df(self.x, *self.args)**2

        self.m_tilde = self.m/(1 - (self.beta_uno**(it+1) ))
        self.s_tilde = self.s/(1 - (self.beta_dos**(it+1) ))

        self.x = self.x - self.rho*(1/(np.sqrt(self.s_tilde) + self.epsilon))*self.m_tilde
        
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