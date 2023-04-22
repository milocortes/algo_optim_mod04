from math import log2, ceil
import random
import numpy as np
import pandas as pd

# Para hacer el muestreo por Latin Hypecube
from scipy.stats.qmc import LatinHypercube,scale

def length_variable(l_sup: int, l_inf: int , precision: int):
    return ceil(log2((l_sup - l_inf)*10**precision))

# Función que obtiene las potencias base 2 de un vector de bits (un individuo)
def to_decimal(dimension,individuo):
    return sum([2**(i) for i in range(dimension-1,-1,-1) ]* np.array(individuo))

# Función que decodifica el vector a un valor real
def binary2real(i_sup, i_inf, dimension, individuo):
    return i_inf+ (to_decimal(dimension, individuo)* ((i_sup-i_inf)/(2**len(individuo)-1)))


class Individuo:
    def __init__(self, f, upper_bound, lower_bound, n_vars, precision, genotipo = []):
        self.f = f
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.n_vars = n_vars
        self.precision = precision
        self.genotipo = genotipo
        self.fenotipo = []
        self.objv = None
        self.aptitud = None
        self.L_genotipo = None

    def construye_genotipo(self):
        acumula_gen = []
        
        for i in range(self.n_vars):
            L_var = length_variable(self.upper_bound[i], self.lower_bound[i], self.precision)
            acumula_gen += [random.randint(0,1)  for j in range(L_var)]

        self.genotipo = acumula_gen

    def decode(self):
        L_total_vars = 0
        for i in range(self.n_vars):
            L_var = length_variable(self.upper_bound[i], self.lower_bound[i], self.precision)

            self.fenotipo.append(
                binary2real(self.upper_bound[i], self.lower_bound[i], L_var, self.genotipo[L_total_vars: L_total_vars + L_var])
            )
            
            L_total_vars += L_var            

        self.L_genotipo = L_total_vars

    def evalua_funcion(self):
        self.objv = self.f(self.fenotipo)

    def calcula_aptitud(self, max_val, min_val, new_max, new_min):
        # scaled_fitness
        y = np.array([new_min, new_max])
        X = np.matrix([[min_val, 1],[max_val, 1]])

        try:
            a,b = np.ravel(X.I @ y)
        except:
            a,b = np.ravel(np.linalg.pinv(X) @ y)
        self.aptitud = a*self.objv + b 
    
    def cruza(self, individuo_cruza):
        
        # Implementación de cruza en un punto
        # Escogemos de forma aleatoria el punto de cruza
        punto_cruza = np.random.choice([i for i in range(self.L_genotipo)])

        return self.genotipo[:punto_cruza] + individuo_cruza.genotipo[punto_cruza:]
    
    def mutacion(self):
        
        self.L_genotipo = len(self.genotipo)

        aleatorio = random.random()
        
        proba_mutacion = 0.1 * (1/self.L_genotipo)
        if aleatorio < proba_mutacion:
            id_swap_gen = np.random.choice([i for i in range(self.L_genotipo)])
            self.genotipo[id_swap_gen] = int(not self.genotipo[id_swap_gen])

def SELECCION(scaled_objv, N, prob_cruza):
    ### Calculamos la probabilidad de selección con el valor de aptitud
    suma = sum(scaled_objv)
    proba_seleccion = [i/suma  for i in scaled_objv]

    ### Obtenemos N parejas para generar la nueva población
    ordena_proba_seleccion = sorted(enumerate(proba_seleccion),key = lambda tup: tup[1], reverse=True)

    suma_acumulada = np.cumsum([v for (k,v) in ordena_proba_seleccion])

    parejas_cruza = []

    for i in range(N):
        pareja = []

        for p in range(2):
            if p == 0:            
                aleatorio = random.random()
                pareja_id = np.argwhere(suma_acumulada >= aleatorio).ravel()[0]
                pareja.append(ordena_proba_seleccion[pareja_id][0])
            else:
                aleatorio = random.random()
                if aleatorio < prob_cruza:
                    aleatorio = random.random()
                    pareja_id = np.argwhere(suma_acumulada >= aleatorio).ravel()[0]
                    pareja.append(ordena_proba_seleccion[pareja_id][0])
                else:
                    mismo_individuo = pareja[0]
                    pareja.append(mismo_individuo)
            
        parejas_cruza.append(pareja)
    
    return parejas_cruza



def genetico_binario(f, N : int, generaciones : int, n_variables : int,  ub : list, lb : list, precision: float, prob_cruza : float):
    '''
    ------------------------------------------
                        
            Genetic Binary Algorithm 
    -------------------------------------------
    # Inputs:
        * f             - Función a minimizar
        * N             - Número de individuos en la población
        * generaciones  - Cantidad máxima de generaciones 
        * n_variables   - Número de variables de decisión
        * ub            - Lista de límites superiores de las variables de decisión
        * lb            - Lista de límites inferiores de las variables de decisión
        * precision     - Precisión de las variables de decisión
        * prob_cruza    - Probabilidad de cruza

    # Output
        * fitness_values - Mejores valores de fitness  
        * best_vector    - Mejor solución encontrada
    '''

    ## Definimos los parámetros del algoritmo genético

    mejor_individuo = 0
    mejor_valor = 1e15
    fitness_values = []

    print(ub)
    #### Inicializamos la población
    poblacion = [ Individuo(f, ub, lb, n_variables, precision) for i in range(N)]

    #### Iniciamos el ciclo evolutivo
    print("Evaluación de la población inicial")

    objv = []

    #### Generamos la población inicial
    for individuo in poblacion:
        # Contruimos el genotipo del individuo
        individuo.construye_genotipo()
        # Decodificamos el genotipo del individuo al dominio del problema (i.e, obtenemos el fenotipo)
        individuo.decode()
        # Evaluamos el fenotipo 
        individuo.evalua_funcion()
        # Guardamos el valor de la función
        objv.append(individuo.objv)

    for it in range(generaciones):
        print("-----------------------------")
        print("-%%%%%%%%%%%%%%%%%%%%%%%%%%%-")
        print("        Generación {}".format(it))
        print("-%%%%%%%%%%%%%%%%%%%%%%%%%%%-")
        print("-----------------------------")

        ### APTITUD de la población
        #### Obtenemos la aptitud de cada individuo
        min_val, max_val = min(objv), max(objv)

        scaled_objv = []

        for individuo in poblacion:
            individuo.calcula_aptitud(max_val, min_val, 0, 100)
            scaled_objv.append(individuo.aptitud) 
        
        ### SELECCIÓN de los individuos que contribuirán a crea la nueva generación
        parejas_cruza = SELECCION(scaled_objv, N, prob_cruza)

        ### Construimos la nueva población con la operación genética de CRUZA
        ##### CRUZA
        nueva_poblacion = []

        for pareja in parejas_cruza:
            
            id_ind_uno, id_ind_dos = pareja
            
            genotipo_cruza = poblacion[id_ind_uno].cruza(poblacion[id_ind_dos])

            nueva_poblacion.append(
                Individuo(f, ub, lb, n_variables, precision, genotipo = genotipo_cruza)
            )

        ##### MUTACIÓN de la población
        for individuo in nueva_poblacion:
            individuo.mutacion()
        
        ##### Actualizamos la nueva población
        poblacion = nueva_poblacion

        #### Evaluamos la nueva población
        objv = [] 
        for individuo in poblacion:
            # Decodificamos el genotipo del individuo al dominio del problema (i.e, obtenemos el fenotipo)
            individuo.decode()
            # Evaluamos el fenotipo 
            individuo.evalua_funcion()
            # Guardamos el valor de la función
            objv.append(individuo.objv)

        #### Identificamos al mejor individuo de la población
        mejor_individuo = objv.index(min(objv))

        #### Actualizamos el mejor valor encontrado
        if objv[mejor_individuo] < mejor_valor:
            mejor_valor = objv[mejor_individuo] 
            mejor_vector = poblacion[mejor_individuo].fenotipo
        
        fitness_values.append(mejor_valor)

    return fitness_values, mejor_vector


# Definimos la clase Particle
class Particle:
    def __init__(self,x,v):
        self.x = x
        self.v = v
        self.x_best = x
        
def PSO(f, pop_size : int, generaciones : int, n_var : int, ub : list, lb : list, α : float, β : float, w : float):
    '''
    ------------------------------------------
                        PSO
    Particle Swarm Optimization
    -------------------------------------------
    # Inputs:
        * f             - Función a minimizar
        * pop_size      - Número de individuos en la población
        * generaciones  - Número de generaciones
        * n_var         - Número de variables de decisión
        * ub            - Lista de límites superiores de las variables de decisión
        * lb            - Lista de límites inferiores de las variables de decisión
        * α             - Parámetro de aprendizaje Social
        * β             - Parámetro de aprendizaje Cognitivo 
        * w             - Velocidad de inercia
        
    # Output
        * x_best         - Mejor solución encontrada
        * fitness_values - Mejores valores de fitness
    '''   
    # LatinHypercube sampling
    # Muestreamos el espacio de búsqueda

    engine = LatinHypercube(d=n_var)
    sample = engine.random(n=pop_size)

    l_bounds = np.array(lb)
    u_bounds = np.array(ub)

    sample_scaled = scale(sample,l_bounds, u_bounds)
    sample_scaled = scale(sample,l_bounds, u_bounds)

    # define particle population
    pob = [Particle(x,np.array([0]*n_var)) for x in sample_scaled]


    
    x_best = pob[0].x_best
    y_best = f(x_best)

    
    # minimum value for the velocity inertia
    w_min = 0.4
    # maximum value for the velocity inertia
    w_max = 0.9

    # Velocidad máxima
    vMax = np.multiply(u_bounds-l_bounds,0.2)
    # Velocidad mínima
    vMin = -vMax

    
    for P in pob:
        y = f(P.x)
        if y < y_best:
            x_best = P.x_best
            y_best = y

    fitness_values = []

    for k in range(generaciones):
        
        print("-----------------------------")
        print("-%%%%%%%%%%%%%%%%%%%%%%%%%%%-")
        print("        Iteración {}".format(k))
        print("-%%%%%%%%%%%%%%%%%%%%%%%%%%%-")
        print("-----------------------------")
        
        for P in pob:
            # Actualiza velocidad de la partícula
            ϵ1,ϵ2 = np.random.uniform(), np.random.uniform()
            P.v = w*P.v + α*ϵ1*(P.x_best - P.x) + β*ϵ2*(x_best - P.x)

            # Ajusta velocidad de la partícula
            index_vMax = np.where(P.v > vMax)
            index_vMin = np.where(P.v < vMin)

            if np.array(index_vMax).size > 0:
                P.v[index_vMax] = vMax[index_vMax]
            if np.array(index_vMin).size > 0:
                P.v[index_vMin] = vMin[index_vMin]

            # Actualiza posición de la partícula
            P.x += P.v

            # Ajusta posición de la particula
            index_pMax = np.where(P.x > u_bounds)
            index_pMin = np.where(P.x < l_bounds)

            if np.array(index_pMax).size > 0:
                P.x[index_pMax] = u_bounds[index_pMax]
            if np.array(index_pMin).size > 0:
                P.x[index_pMin] = l_bounds[index_pMin]

            # Evaluamos la función
            y = f(P.x)

            if y < y_best:
                x_best = np.copy(P.x_best)
                y_best = y
            if y < f(P.x_best):
                P.x_best = np.copy(P.x)
            

            # Actualizamos w

            w = w_max - k * ((w_max-w_min)/generaciones)

        fitness_values.append(y_best)

    return fitness_values ,x_best