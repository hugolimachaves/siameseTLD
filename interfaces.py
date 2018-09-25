import numpy as np
from ctypes import *
from sklearn.neighbors import KNeighborsClassifier

shared_library = CDLL('TLD/bin/Debug/libTLD.so')

positive_obj_model    = []
negative_obj_model    = []
feature_pos_obj_model = []
feature_neg_obj_model = []

SIZE_ARRAY = 32
LAST_ADDED = -1
SIZE_DESCRIPTOR = 255

def convertSimilatiry(siameseDistance):
    return 1 / (siameseDistance+1) # retorna a distancia no TLD
    
def getLength(element): # verifica o tamanho total de elementos em uma estrutura de dados de dimensoes arbitrárias
    if isinstance(element, list):
        return sum(([getLength(i) for i in element]))
    return 1

def getDescriptor(bb):
    descriptor = []
    #TODO Estamos colocando apenas um place holder. A funcao depende da analise do tracker siameseFC no python
    for _ in range(SIZE_DESCRIPTOR):
        descriptor.append(float(np.random.randn()))
    return descriptor

# passa  as deep Features dos candidatos para o presente frame conjuntamente
# com o modelo positivo(default) ou negativo
def distCandidatesToTheModel(deep_features_candidates, isPositive=True):
    #Usa os seguintes parametros globais:  feature_pos_obj_model, feature_neg_obj_model  
    features = []

    if isPositive: # modelo positivo do object model
        positiveLabel = [1 for i in feature_pos_obj_model]
        labels = positiveLabel
        #mod 

        #features = feature_pos_obj_model    
        #~mod
    
    else: # modelo negativo do object model
        negativeLabel = [0 for i in feature_neg_obj_model] 
        labels = negativeLabel
        #mod
        #features = feature_neg_obj_model
    
    
    print(features)
    knn_1 = KNeighborsClassifier(n_neighbors=1)

    #list_features = []
    #list_features.append(features)
    list_features = deep_features_candidates
    print('list_features len printando:',len(list_features))
    knn_1.fit(list_features, labels)
    
    distances = []
    positions = []
    for candidate in deep_features_candidates: # pega a menor distancia para cada candidato na lista deep_features_candidate
        list_candidate = []
        list_candidate.append(candidate)
        dist,position = knn_1.kneighbors(list_candidate, n_neighbors=1, return_distance=True)
        distances.append(dist)
        positions.append(position)
        # example: neigh.kneighbors([[1., 1., 1.]])
        # pode das errado porque a documentacao mostra um array de array

    return distances, positions # retorna a menor distancia em relação ao modelo, é uma lista pois sao varios candidatos e tambem  a posição no vetor

# passo uma lista de bb dos candidatos e uma bb do tracker (lista de lista)
def detSimilarity(candidates, bb_tracker, is_neg_empty=False, is_pos_empty=False, is_candidates_empty=False, is_bb_tracker_empty=False):
    # sera utilizado - (sao globais): positive_obj_model, negative_obj_model
    if(not is_neg_empty):
        for bb in negative_obj_model[LAST_ADDED]:
            descriptor = getDescriptor(bb)
            feature_neg_obj_model.append(descriptor) # o tamanho e equivalente ao numero de descritores negativos

        assert(getLength(feature_pos_obj_model) == getLength(positive_obj_model)), 'o tamanho do object Model difere do tamanho dos descritores'

    if(not is_pos_empty):
        for bb in positive_obj_model[LAST_ADDED]:
            descriptor = getDescriptor(bb)
            feature_pos_obj_model.append(descriptor)  # o tamanho e equivalente ao numero de descritores positivos

        assert(getLength(feature_neg_obj_model) == getLength(negative_obj_model)), 'o tamanho do object Model difere do tamanho dos descritores'

    if(not is_candidates_empty):
        #deep_features_candidates = []
        for bb in candidates:
            #TODO fazer o descritor
            descriptor = getDescriptor(bb)
            descriptor = [[i] for i in descriptor]
            print('descritor'.center(50,'*'))
            print('O descritor provisorio e: ', descriptor)
            print('~descritor'.center(50,'*'))
        deep_features_candidates = descriptor
        print('Info deep_features_candidates: ',  deep_features_candidates, '\n tipo: ', type(deep_features_candidates) )
        positive_distances_candidates = distCandidatesToTheModel(deep_features_candidates, isPositive=True)
        negative_distances_candidates = distCandidatesToTheModel(deep_features_candidates, isPositive=False)

    if(not is_bb_tracker_empty):
        for bb in bb_tracker:
            descriptor = getDescriptor(bb)
            feature_tracker.append(descriptor)

        positive_distances_tracker = distCandidatesToTheModel(feature_tracker, isPositive=True)
        negative_distances_tracker = distCandidatesToTheModel(feature_tracker, isPositive=False)

    return 2 #positive_distances_candidates, negative_distances_candidates,
             #positive_distances_tracker,    negative_distances_tracker



#'frame' se refere ao numero do frame que esta sendo processado no codigo .py
def interface1(frame): 
    '''
    codigo de execucao do c/c++ aqui!
     
    Parametros: (frame ou void)
 
    Retorno do codigo:
    1) lista com as posicoes extraidas, 'lista1'. pode ser uma lista de estruturas de 5 valores,
        (4 referentes a localizacao do frame, e um indicando se o modelo eh positivo-1- ou negativo-0)
        Caso precise retornar um numero fixo de valores, entre em contato comigo. A principio, se precisar retornar
        um vetor de tamanho fixo, adote um vetor de 100 estruturas de 5 posicoes. Onde nao ha nada preenchido, coloque
        valores negativos, como: -1
    2)Frame ao qual foi processada as informacoes, que alimentara a variavel: retorno_frame.
    Vaviavel necessaria para garantir o processamento do mesmo frame.
    '''

    arry_size = 100 # Example

    parameters_path = "/home/hugo/Documents/Mestrado/codigoKevyn2/dataset/exemplo/01-Light_video00001/parameters.yml"
    parameters_path = parameters_path.encode('utf-8')

    retorno_frame = c_int()

    size_candidates = c_int()
    size_positive   = c_int()
    size_negative   = c_int()
    size_bb_tracker = c_int()

    array_bb_candidates         = [-1] * arry_size
    array_object_model_positive = [-1] * arry_size
    array_object_model_negative = [-1] * arry_size

    array_bb_candidates         = (c_float * arry_size) (*array_bb_candidates)
    array_object_model_positive = (c_float * arry_size) (*array_object_model_positive)
    array_object_model_negative = (c_float * arry_size) (*array_object_model_negative)

    array_bb_tracker = [-1] * 4
    array_bb_tracker = (c_float * 4) (*array_bb_tracker)

    print(type(parameters_path))
    shared_library.initializer_TLD(parameters_path)

    shared_library.TLD_function_1(byref(retorno_frame), array_bb_candidates, byref(size_candidates),
                   array_object_model_positive, byref(size_positive), 
                   array_object_model_negative, byref(size_negative),
                   array_bb_tracker, byref(size_bb_tracker))

    candidates = []
    bb_tracker = []

    is_neg_empty        = True
    is_pos_empty        = True
    is_candidates_empty = True
    is_bb_tracker_empty = True

    if(size_positive.value   is not 0):
        print('\n\nOs valores do object model ++ :')
        bb_list = []
        bb_pos = []
        for i in range(size_positive.value):
            bb_pos.append(array_object_model_positive[i])

            if(i%4==0 and i is not 0):
                bb_pos.append(frame)
                bb_list.append(bb_pos)
                bb_pos = []
                print()
                
            print(str(array_object_model_positive[i])+' ',end='')

        positive_obj_model.append(bb_list)
        is_pos_empty = False

    if(size_negative.value is not 0):
        print('\n\nOs valores do object model -- :')
        bb_list = []
        bb_pos = []
        for i in range(size_negative.value):
            bb_pos.append(array_object_model_negative[i])

            if(i%4==0 and i is not 0):
                bb_pos.append(frame)
                bb_list.append(bb_pos)
                bb_pos = []
                print()
                
            print(str(array_object_model_negative[i])+' ',end='')

        negative_obj_model.append(bb_list)
        is_neg_empty = False

    if(size_candidates.value is not 0):
        print('\n\nOs valores dos candidatos sao:')
        bb_list = []
        bb_pos = []
        for i in range(size_candidates.value):
            bb_pos.append(array_bb_candidates[i])

            if(i%4==0 and i is not 0):
                bb_pos.append(frame)
                bb_list.append(bb_pos)
                bb_pos = []
                print()
                
            print(str(array_bb_candidates[i])+' ',end='')

        candidates.append(bb_list)
        is_candidates_empty = False

    if(size_bb_tracker.value is not 0):
        print('\n\nOs valores da bb do Tracker:\n')
        bb_pos = []
        for i in range(size_bb_tracker.value):
            bb_pos.append(array_bb_tracker[i])
            print(str(array_bb_tracker[i])+' ',end='')

        bb_tracker.append(bb_list)
        is_bb_tracker_empty = False

    
    print('\nFrame de entrada: '+ str(frame)+ ' Frame de retorno: ' + str(retorno_frame.value))
    assert (frame == retorno_frame.value), "Conflito nos frames"

    # candidates[ N ][ 5 ]
    #   - N: Numero de bb no frame
    #   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

    # positive_obj_model[ P ][ N ][ 5 ]
    #   - P: Numero de frames que retornaram bb ate o momento
    #   - N: Numero de bb do frame no p-esimo frame
    #   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

    # negative_obj_model[ P ][ N ][ 5 ]
    #   - P: Numero de frames que retornaram bb ate o momento
    #   - N: Numero de bb do frame no p-esimo frame
    #   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

    # bb_tracker[ 1 ][ 5 ]
    #   - 1: Para ser considerado uma lista no algoritmos
    #   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

    # passa uma lista de bb dos candidatos retornado pelo TLD e passa uma bb retornado pelo Tracker no TLD

    retorno = detSimilarity(candidates, bb_tracker, is_neg_empty, is_pos_empty, is_candidates_empty, is_bb_tracker_empty)

    shared_library.TLD_function_2(byref(retorno_frame))

    assert (frame == retorno_frame.value), "Conflito nos frames"
    



    return lista1

interface1(2)
 
#'deepDescriptor' eh o descritor que sera passado para o codigo C. eh um descritor de 128/256 floats.
def interface2(deepDescriptor,frame):

    '''
    codigo de execucao do c/c++ aqui!
     
    Parametros( deepDescriptor e (frame ou void)
 
    1)retorna 'objectModel' que eh uma lista de posicoes do dos modelos de objetos detectados nesse, e somente nesse, frame,
    Onde a estrutura tem 5 elementos, 4 para posicoes e 1 para indicar se esse posicao eh positiva ou negativa. Retornne tambehm
    o numero do frame processado, para verificacao do assert.
    Caso precise retorna uma lista de tamanho fixo (um vetor), siga as recomendacoes do item 1) do comentario para interface1.
    2)Frame ao qual foi processada as informacoes, que alimentara a variavel: 'retornoFrame'.
    Vaviavel necessaria para garantir o processamento do mesmo frame.
    '''
    assert (frame==retornoFrame), "Conflito nos frames"
    return objectModel

