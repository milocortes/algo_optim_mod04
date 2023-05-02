from math import log2, ceil
import random
import numpy as np
import pandas as pd


from multiprocessing import Pool,cpu_count, Value, Array
import math
import ctypes

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
        * mejor_vector   - Mejor solución encontrada
        * mejor_valor    - Mejor valor de la función objetivo
        * fitness_values - Mejores valores de fitness  
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

        if it % 20 == 0:
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

    return mejor_vector, mejor_valor, fitness_values


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
        * y_best         - Mejor valor de la función objetivo
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
        
        if k % 20 == 0:
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

    return x_best, y_best, fitness_values 


    
def init_pso(gbest_val_arg, gbest_pos_arg, position_arg, velocity_arg, pbest_val_arg, 
         pbest_pos_arg,f_optim,α_arg,β_arg,w_arg,vMax_arg,vMin_arg,
         u_bounds_arg,l_bounds_arg):
    global gbest_val
    global gbest_pos
    global position
    global velocity
    global pbest_val
    global pbest_pos
    global f
    global α
    global β
    global w
    global vMax
    global vMin
    global u_bounds
    global l_bounds

    gbest_val = gbest_val_arg
    gbest_pos = gbest_pos_arg
    position = position_arg
    velocity = velocity_arg
    pbest_val = pbest_val_arg
    pbest_pos = pbest_pos_arg
    f = f_optim
    
    # Cognitive scaling parameter
    α = α_arg
    # Social scaling parameter
    β = β_arg
    
    # velocity inertia
    w = w_arg
    
    vMax = vMax_arg
    vMin = vMin_arg
    u_bounds = u_bounds_arg
    l_bounds = l_bounds_arg
    
def evalua_f_pso(i):    
    # Actualiza velocidad de la partícula
    ϵ1,ϵ2 = np.random.RandomState().uniform(), np.random.RandomState().uniform()
    with gbest_pos.get_lock():
        velocity[i] = w.value*velocity[i] + α*ϵ1*(pbest_pos[i] -  position[i]) + β*ϵ2*(np.array(gbest_pos[:]) - position[i])

            
    # Ajusta velocidad de la partícula
    index_vMax = np.where(velocity[i] > vMax)
    index_vMin = np.where(velocity[i] < vMin)

    if np.array(index_vMax).size > 0:
        velocity[i][index_vMax] = vMax[index_vMax]
    if np.array(index_vMin).size > 0:
        velocity[i][index_vMin] = vMin[index_vMin]

    # Actualiza posición de la partícula
    position[i] = position[i] + velocity[i] 

    # Ajusta posición de la particula
    index_pMax = np.where(position[i] > u_bounds)
    index_pMin = np.where(position[i] < l_bounds)

    if np.array(index_pMax).size > 0:
        position[i][index_pMax] = u_bounds[index_pMax]
    if np.array(index_pMin).size > 0:
        position[i][index_pMin] = l_bounds[index_pMin]

    # Evaluamos la función
    y = f(position[i])
    with gbest_val.get_lock():
        if y < gbest_val.value:
            with gbest_pos.get_lock(): 
                gbest_pos[:] = np.copy(position[i]) 
                pbest_pos[i] = np.copy(position[i])
                gbest_val.value = y
        if y < pbest_val[i]:
            pbest_pos[i] = np.copy(position[i])

def PSO_par(f_cost, pop_size, max_iters, ub, lb, α, β, w, w_max, w_min):

    '''
    ------------------------------------------
                        PSO_par
    Particle Swarm Optimization
    -------------------------------------------
    # Inputs:
        * f_cost        - Función a minimizar
        * pop_size      - Número de individuos en la población
        * max_iters     - Número de generaciones
        * n_var         - Número de variables de decisión
        * ub            - Lista de límites superiores de las variables de decisión
        * lb            - Lista de límites inferiores de las variables de decisión
        * α             - Parámetro de aprendizaje Social
        * β             - Parámetro de aprendizaje Cognitivo 
        * w             - Velocidad de inercia
        * w_max         - Velocidad máxima de inercia
        * w_min         - Velocidad mínima de inercia
        
    # Output
        * x_best         - Mejor solución encontrada
        * y_best         - Mejor valor de la función objetivo
        * fitness_values - Mejores valores de fitness
    '''   

    # Tamaño de la población
    n = pop_size
    maxiter = max_iters
    # Número de variables
    n_var = len(lb)

    # Cognitive scaling parameter
    α = α
    # Social scaling parameter
    β = β

    # velocity inertia
    w = Value(ctypes.c_double,w)
    # minimum value for the velocity inertia
    w_min = w_min
    # maximum value for the velocity inertia
    w_max = w_max

    # Usamos Latin Hypercube Sampling para muestrear puntos en el espacio de búsqueda
    engine = LatinHypercube(d=n_var)
    sample = engine.random(n=n)

    # Definimos los límites superiores e inferiores para las variables de decisión
    l_bounds = np.array(lb)
    u_bounds = np.array(ub)

    # Creamos un arreglo compartido para el vector de limites superiores
    mp_l_bounds = Array(ctypes.c_double,l_bounds)
    # Creamos un nuevo arreglo de numpy usando el arreglo compartido
    np_l_bounds = np.frombuffer(mp_l_bounds.get_obj(), ctypes.c_double) 

    # Creamos un arreglo compartido para el vector de limites superiores
    mp_u_bounds = Array(ctypes.c_double,u_bounds)
    # Creamos un nuevo arreglo de numpy usando el arreglo compartido
    np_u_bounds = np.frombuffer(mp_u_bounds.get_obj(), ctypes.c_double) 

    # Velocidad máxima
    vMax = np.multiply(u_bounds-l_bounds,0.2)
    # Creamos un arreglo compartido para el vector de velocidad máxima
    mp_vMax = Array(ctypes.c_double,vMax) 
    # Creamos un nuevo arreglo de numpy usando el arreglo compartido
    np_vMax = np.frombuffer(mp_vMax.get_obj(), ctypes.c_double) 

    # Velocidad mínima
    vMin = -vMax
    # Creamos un arreglo compartido para el vector de velocidad máxima
    mp_vMin = Array(ctypes.c_double,vMin) 
    # Creamos un nuevo arreglo de numpy usando el arreglo compartido
    np_vMin = np.frombuffer(mp_vMin.get_obj(), ctypes.c_double) 


    # Escalamos los valores muestreados de LHS
    sample_scaled = scale(sample,l_bounds, u_bounds)

    # Creamos un arreglo compartido para el vector de velocidad
    mp_vel = Array(ctypes.c_double,n*n_var)
    # Creamos un nuevo arreglo de numpy usando el arreglo compartido
    vel = np.frombuffer(mp_vel.get_obj(), ctypes.c_double)
    # Convertimos a un arreglo 2-dimensional
    vel_resh = vel.reshape((n,n_var))

    # Creamos un arreglo compartido para el vector de posición
    mp_pos = Array(ctypes.c_double,n*n_var)
    # Creamos un nuevo arreglo de numpy usando el arreglo compartido
    pos = np.frombuffer(mp_pos.get_obj(), ctypes.c_double)
    # Convertimos a un arreglo 2-dimensional
    pos_resh = pos.reshape((n,n_var))
    # Inicializamos el vector de posición con el vector muestreado por LHS
    for i,v in enumerate(sample_scaled):
        pos_resh[i] = v

    # Mejor valor global (compartido) de la función objetivo
    gbest_val = Value(ctypes.c_double,math.inf)
    # Mejor vector de posición global (compartido)
    gbest_pos = Array(ctypes.c_double, sample_scaled[0])

    # Mejor valor para cada partícula
    pbest_val_arg = Array(ctypes.c_double, [math.inf]*n )

    # Mejor vector de posición individual para cada partícula
    pbest_pos_mp = Array(ctypes.c_double,n*n_var)
    # Creamos un nuevo arreglo de numpy usando el arreglo compartido
    pbest_pos = np.frombuffer(pbest_pos_mp.get_obj(), ctypes.c_double)
    # Convertimos a un arreglo 2-dimensional
    pbest_pos_arg = pbest_pos.reshape((n,n_var))
    # Inicializamos el vector de posición con el vector muestreado por LHS
    for i,v in enumerate(sample_scaled):
        pbest_pos_arg[i] = v

    p = Pool(processes = int(cpu_count()),initializer=init_pso,
            initargs=(gbest_val,gbest_pos,pos_resh, vel_resh, 
                      pbest_val_arg, pbest_pos_arg, f_cost,α,β,w,
                      np_vMax,np_vMin,np_u_bounds,np_l_bounds,))

    fitness_values = []
    for k in range(maxiter):
        p.map(evalua_f_pso, range(n))
        print("It {}  gbest_val {}".format(k, gbest_val.value))

        # Actualizamos w
        w.value = w_max - k * ((w_max-w_min)/maxiter)

        fitness_values.append(gbest_val.value)

    return gbest_pos[:], gbest_val.value, fitness_values