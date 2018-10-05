# Suofei ZHANG, 2017.
import os
import tensorflow as tf
import numpy as np
import glob
import cv2
import scipy.io as sio
import time
from PIL import Image
import utils
from siamese_net import SiameseNet
from parameters import configParams
from ctypes import *
from sklearn.neighbors import KNeighborsClassifier
'''from interfaces.py'''

DEBUG_PRINT_ARRAY = True
DIM_DESCRIPTOR = 256
ONE_DIMENSION = 1
ATOMIC_SIZE = 87
SIZE_ARRAY = 32
LAST_ADDED = -1
SIZE_DESCRIPTOR = 256
HULL_SIZE = 4
WINDOW_SIZE = 40 # Example bb_list
OBJECT_MODEL_SIZE = 400 # Example
ARRAY_SIZE = 500 # Example
RGB = 3
POSICAO_PRIMEIRO_FRAME = 0
POSICAO_SEGUNDO_FRAME = 1
PRIMEIRO_FRAME = 1
SEGUNDO_FRAME =2
ULTIMO_FRAME = 354

YML_FILE_NAME = 'parameters.yml'
PARAMETERS_PATH = os.path.join(os.getcwd(),'dataset/exemplo/01-Light_video00001',YML_FILE_NAME)


shared_library = CDLL('TLD/bin/Debug/libTLD.so')



'''
 good_windows_hull[ N ][ 5 ]
   - N: Numero de bb no frame
   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

 positive_obj_model[ P ][ N ][ 5 ]
   - P: Numero de frames que retornaram bb ate o momento
   - N: Numero de bb do frame no p-esimo frame
   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

 negative_obj_model[ P ][ N ][ 5 ]
   - P: Numero de frames que retornaram bb ate o momento
   - N: Numero de bb do frame no p-esimo frame
   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

 good_windows[ P ][ N ][ 5 ]
   - P: Numero de frames que retornaram bb ate o momento
   - N: Numero de bb do frame no p-esimo frame
   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)
'''

class Generation:
	
	def __init__(self,opts,siamiseNetWorkLocal):
		self.minimumSiameseNetPlaceHolder = tf.placeholder(tf.float32, [ONE_DIMENSION, opts['minimumSize'], opts['minimumSize'], RGB])
		tf.convert_to_tensor(False, dtype='bool', name='is_training')
		isTrainingOp = tf.convert_to_tensor(False, dtype='bool', name='is_training')
		self.zMinimumPreTrained =siamiseNetWorkLocal.buildExemplarSubNetwork(self.minimumSiameseNetPlaceHolder,opts,isTrainingOp)
		self.tensorFlowSession = tf.Session()
		tf.initialize_all_variables().run(session=self.tensorFlowSession)
		

	#passando Bounding Box no formato X,Y,W,H, retornando left, top, right, botton
	def get_image_cropped(self, img, bb): # imagemCurrentFrame PIL
		left	= round(bb[0] - (bb[2]/2))
		top	 = round(bb[1] - (bb[3])/2)
		right   = round(bb[2] + left)
		bottom  = round(bb[3] + top)

		cropped = img.crop([left,top,right,bottom])
		return cropped

	def getDescriptor(self,bb,imageSource): # zMinimumFeatures = sess.run(zMinimumPreTrained, feed_dict={minimumSiameseNetPlaceHolder: zCropMinimum})
		imImageSource = self.get_image_cropped(imageSource,bb)
		neoImageSource = imImageSource.resize((ATOMIC_SIZE,ATOMIC_SIZE))
		npImageSource = np.array(neoImageSource)
		npImageSource = npImageSource.reshape(1,npImageSource.shape[0],npImageSource.shape[1],3)
		zMinimumFeatures = self.tensorFlowSession.run(self.zMinimumPreTrained, feed_dict={self.minimumSiameseNetPlaceHolder: npImageSource})
		
		return zMinimumFeatures
'''
bad_windows  = []
good_windows = []
good_windows_hull   = []
positive_obj_model	= []
negative_obj_model	= []
feature_pos_obj_model = []
feature_neg_obj_model = []
'''
class DeepDescription:
	positive_obj_model_bb		 = []
	negative_obj_model_bb 		 = []
	good_windows_bb 	 		 = []
	good_windows_hull_bb 		 = []
	tracker_bb					 = []
	__candidates_bb 			 = []

	positive_obj_model_features  = []
	negative_obj_model_features  = []
	good_windows_features 		 = []
	good_windows_hull_features   = []
	tracker_features			 = []
	__candidates_features 		 = []

	#TODO Colocar privado
	positive_distances_candidates  = []
	negative_distances_candidates  = []
	positive_similarity_candidates = []
	negative_similarity_candidates = []

	#TODO Colocar privado
	positive_distances_tracker_candidate  = []
	negative_distances_tracker_candidate  = []
	positive_similarity_tracker_candidate = []
	negative_similarity_tracker_candidate = []
	
	__currentFrame = 0

	def __init__(self):
		self.__currentFrame = 0

	def setCandidates(self,candBB,candFeat,currentFrame):
		self.__currentFrame = currentFrame

		self.__candidates_bb = []
		self.__candidates_bb = candBB

		self.__candidates_features = []
		self.__candidates_features = candFeat

	def getCandidates(self,currentFrame):
		if (currentFrame is self.__currentFrame):
			return self.__candidates_bb, self.__candidates_features

		self.__currentFrame = currentFrame
		return [], []
	
generalDescriptor = DeepDescription()

def convertSimilatiry(siameseDistance):
	#print('Siamese distance: ', siameseDistance)
	return 1.0 / (siameseDistance + 1.0) # retorna a distancia no TLD
	
def getLength(element): # verifica o tamanho total de elementos em uma estrutura de dados de dimensoes arbitrarias
	if isinstance(element, list):
		return sum(([getLength(i) for i in element]))
	return 1

# passa  as deep Features dos candidatos para o presente frame conjuntamente
# com o modelo positivo(default) ou negativo
def distCandidatesToTheModel(deep_features_candidates, isPositive=True):
	#Usa os seguintes parametros globais:  feature_pos_obj_model, feature_neg_obj_model  
	features = []

	if isPositive: # modelo positivo do object model
		positiveLabel = [1 for i in feature_pos_obj_model]
		labels = np.asarray(positiveLabel)
		features = np.asarray(feature_pos_obj_model)
	
	else: # modelo negativo do object model
		negativeLabel = [0 for i in feature_neg_obj_model] 
		labels = np.asarray(negativeLabel)
		features = np.asarray(feature_neg_obj_model)
	
	distances = []
	positions = []

	#print('features.size ', features.size)
	if (features.size is not 0):
		knn_1 = KNeighborsClassifier(n_neighbors=1)
		listFeatures = [bb for frame in features for bb in frame]
		knn_1.fit(listFeatures, labels)
		
		for candidate in deep_features_candidates: # pega a menor distancia para cada candidato na lista deep_features_candidate
			list_candidate = np.asarray(candidate)
			dist,position = knn_1.kneighbors(list_candidate, n_neighbors=1, return_distance=True)
			distances.append(dist[0][0])
			positions.append(position)
			# example: neigh.kneighbors([[1., 1., 1.]])
			# pode das errado porque a documentacao mostra um array de array

	return distances # retorna a menor distancia em relacao ao modelo, eh uma lista pois sao varios candidatos e tambem  a posicao no vetor

	#return distances # retorna a menor distancia em relacao ao modelo, eh uma lista pois sao varios candidatos e tambem  a posicao no vetor

# passo duas features e calcula a distancia euclidiana entre elas
def detSimilarity(feature_a, feature_b):
	dist = 0
	#print('featura a: ', feature_a)
	#print('feature b: ',feature_b)
	if len(feature_a.shape) > 2:
		feature_a = feature_a.reshape((max(feature_a.shape),ONE_DIMENSION))
	if len(feature_b.shape) > 2:
		feature_b = feature_b.reshape((max(feature_b.shape),ONE_DIMENSION))
	for a, b in zip(feature_a, feature_b):
		dist += (a - b) ** 2

	print('feature_a.shape', feature_a.shape)
	print('dist in desSimilarity', np.sqrt(dist))
	return np.sqrt(dist)

	'''
	positive_distances_candidates = []
	negative_distances_candidates = []
	positive_distances_tracker = []
	negative_distances_tracker = []

	positive_distances_candidates = distCandidatesToTheModel(deep_features_candidates, isPositive=True)
	negative_distances_candidates = distCandidatesToTheModel(deep_features_candidates, isPositive=False)

	positive_distances_tracker = distCandidatesToTheModel(feature_tracker, isPositive=True)
	negative_distances_tracker = distCandidatesToTheModel(feature_tracker, isPositive=False)

	positive_siam_sim_cand = [convertSimilatiry(distancia) for distancia in positive_distances_candidates]
	negative_siam_sim_cand = [convertSimilatiry(distancia) for distancia in negative_distances_candidates]
	positive_siam_sim_bb_tracker = [convertSimilatiry(distancia) for distancia in positive_distances_tracker]
	negative_siam_sim_bb_tracker = [convertSimilatiry(distancia) for distancia in negative_distances_tracker]

	return positive_siam_sim_cand, negative_siam_sim_cand, positive_siam_sim_bb_tracker, negative_siam_sim_bb_tracker
	'''

def read_data(array, array_size, frame, name=0):
	bb_list = []
	is_empty = True
	
	if DEBUG_PRINT_ARRAY and name is not 0:
		if(name == 1):
			print('\n\tNegativo ', end='')
		if(name == 2):
			print('\n\tPositivo ', end='')
		if(name == 3):
			print('\n\tCandidatos ', end='')
		if(name == 4):
			print('\n\tBoundding Box do tracker ', end='')

		print('array: ',end='')

	if(array_size is not 0):
		bb_pos = []
		for i in range(array_size):
			bb_pos.append(array[i])

			if(i%4==0 and i is not 0):
				bb_pos.append(frame)
				bb_list.append(bb_pos)
				bb_pos = []

			if DEBUG_PRINT_ARRAY and name is not 0:
				if i%4 == 0:
					print('[', end='')
					print(array[i],end=''))

				if i%4 == 3:
					print(array[i],end='')
					print('] ', end='')
				else:
					print(array[i],', ',end='')

		bb_pos.append(frame)
		bb_list.append(bb_pos)
		bb_pos = []
		is_empty = False

	return bb_list, is_empty


#'frame' se refere ao numero do frame que esta sendo processado no codigo .py
#/home/hugo/Documents/Mestrado/codigoSiameseTLD/siameseTLD/dataset/exemplo/01-Light_video00001/parameters.yml
def init_TLD_in_siameseFC(generated, imgs_pil, frame=1): 
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

	parameters_path = PARAMETERS_PATH.encode('utf-8')
	retorno_frame = c_int() 	# numero do frame atual
	size_negative = c_int(0)	# tamanho do vetor array objectModel negativo
	size_positive = c_int(0) 	# tamanho do vetor array objectModel positivo
	size_good_windows = c_int(0) 		# tamanho do vetor array good windows
	size_good_windows_hull = c_int(0)	# tamanho do vetor array good_windows_hull (que e sempre 4)

	array_good_windows		  = [-1] * WINDOW_SIZE 			# placeholders
	array_good_windows_hull	  = [-1] * HULL_SIZE 			# placeholders
	array_object_model_negative = [-1] * OBJECT_MODEL_SIZE 	# placeholders
	array_object_model_positive = [-1] * OBJECT_MODEL_SIZE 	# placeholders

	# fazendo a alocacao dos vetores, (*array_good_windows) --> posicao de memoria do vetor
	array_good_windows 			= (c_float * WINDOW_SIZE) (*array_good_windows) 
	array_good_windows_hull	 	= (c_float * 4) (*array_good_windows_hull)
	array_object_model_negative = (c_float * OBJECT_MODEL_SIZE) (*array_object_model_negative) 
	array_object_model_positive = (c_float * OBJECT_MODEL_SIZE) (*array_object_model_positive) 

	shared_library.initializer_TLD(parameters_path, byref(retorno_frame), 
								   array_object_model_positive, byref(size_positive), 
								   array_object_model_negative, byref(size_negative),
								   array_good_windows, byref(size_good_windows),
								   array_good_windows_hull,  byref(size_good_windows_hull))
	
	print('\nFrame de entrada: '+ str(frame)+ ' Frame de retorno: ' + str(retorno_frame.value) + '\n')
	assert (frame == retorno_frame.value), "Conflito nos frames"

	bb_list_negativo, is_neg_empty = read_data(array_object_model_negative, size_negative.value, frame, 1)
	bb_list_positivo, is_pos_empty = read_data(array_object_model_positive, size_positive.value, frame, 2)
	bb_list_good_window, is_good_window_empty = read_data(array_good_windows, size_good_windows.value, frame)
	bb_good_windows_hull, is_good_hull_empty = read_data(array_good_windows_hull, size_good_windows_hull.value, frame)

	addModel(generated, bb_list_negativo,	  generalDescriptor.negative_obj_model_bb, generalDescriptor.negative_obj_model_features, imgs_pil[POSICAO_PRIMEIRO_FRAME])
	addModel(generated, bb_list_positivo,	  generalDescriptor.positive_obj_model_bb, generalDescriptor.positive_obj_model_features, imgs_pil[POSICAO_PRIMEIRO_FRAME]) 
	addModel(generated, bb_list_good_window,  generalDescriptor.good_windows_bb,	   generalDescriptor.good_windows_features,		  imgs_pil[POSICAO_PRIMEIRO_FRAME])
	addModel(generated, bb_good_windows_hull, generalDescriptor.good_windows_hull_bb,  generalDescriptor.good_windows_hull_features,  imgs_pil[POSICAO_PRIMEIRO_FRAME])

	#print('\n\n\n##############      \n',generalDescriptor.good_windows_hull_features)
	#return bb_list_negativo, is_neg_empty, bb_list_positivo, is_pos_empty, bb_list_good_window, is_good_window_empty, bb_good_windows_hull, is_good_hull_empty

def TLD_parte_1(generated, imgs_pil, frame):
	retorno_frame = c_int()

	size_candidates = c_int()
	size_positive   = c_int()
	size_negative   = c_int()
	size_bb_tracker = c_int()

	array_bb_candidates			= [-1] * ARRAY_SIZE
	array_object_model_positive = [-1] * ARRAY_SIZE
	array_object_model_negative = [-1] * ARRAY_SIZE

	array_bb_candidates			= (c_float * ARRAY_SIZE) (*array_bb_candidates)
	array_object_model_positive = (c_float * ARRAY_SIZE) (*array_object_model_positive)
	array_object_model_negative = (c_float * ARRAY_SIZE) (*array_object_model_negative)

	array_bb_tracker = [-1] * 4
	array_bb_tracker = (c_float * 4) (*array_bb_tracker)

	shared_library.TLD_function_1(byref(retorno_frame), array_bb_candidates, byref(size_candidates),
									array_object_model_positive, byref(size_positive), 
									array_object_model_negative, byref(size_negative),
									array_bb_tracker, byref(size_bb_tracker))
	
	print('\nFrame de entrada: '+ str(frame)+ ' Frame de retorno: ' + str(retorno_frame.value))
	assert (frame == retorno_frame.value), "Conflito nos frames"

	candidates = []
	bb_tracker = []

	is_candidates_empty = True
	is_bb_tracker_empty = True

	bb_list_negativo, is_neg_empty = read_data(array_object_model_negative, size_negative.value, frame, 1)
	bb_list_positivo, is_pos_empty = read_data(array_object_model_positive, size_positive.value, frame, 2)
	bb_list_candidate, is_candidates_empty = read_data(array_bb_candidates, size_candidates.value, frame, 3)
	bb_single_element_tracker, is_bb_tracker_empty = read_data(array_bb_tracker, size_bb_tracker.value, frame, 4)

	'''
	 candidates[ N ][ 5 ]
	   - N: Numero de bb no frame
	   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

	 bb_tracker[ 1 ][ 5 ]
	   - 1: Para ser considerado uma lista no algoritmos
	   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)
	'''

	posicao = frame - 1

	addModel(generated, bb_list_negativo, generalDescriptor.negative_obj_model_bb, generalDescriptor.negative_obj_model_features, imgs_pil[posicao])
	addModel(generated, bb_list_positivo, generalDescriptor.positive_obj_model_bb, generalDescriptor.positive_obj_model_features, imgs_pil[posicao])
	addModel(generated, bb_single_element_tracker, generalDescriptor.tracker_bb, generalDescriptor.tracker_features, imgs_pil[posicao])
	
	list_feature = []
	list_bb 	 = []
	addModel(generated, bb_list_candidate, list_bb, list_feature, imgs_pil[posicao])
	generalDescriptor.setCandidates(list_bb, list_feature, frame)

	# Calculo das distancias e similaridades para os candidatos	

	_ , features_candidates = generalDescriptor.getCandidates(frame)

	'''
	print('features_candidates: ',features_candidates)
	print('generalDescriptor.positive_obj_model_features ', generalDescriptor.positive_obj_model_features)
	print('Tam de features_candidates: ',len(features_candidates))
	print('Tam de generalDescriptor.positive_obj_model_features: ',len(generalDescriptor.positive_obj_model_features))
	'''

	for candidate in features_candidates:
		#print('features_candidates: ',features_candidates)
		distances_candidate = []
		for positive in generalDescriptor.positive_obj_model_features:
			cand_np_array = np.asarray(candidate)
			#print('Candidate: ', candidate)
			#print('NP Candidate: ', cand_np_array)
			#print('Shape Candidate: ', cand_np_array.shape)
			dist = detSimilarity(candidate, positive)
			distances_candidate.append(dist)	# Lista das distancias em relacao as features positivas
			
		generalDescriptor.positive_distances_candidates.append(distances_candidate)	# Lista das distancias para cada candidato
		generalDescriptor.positive_similarity_candidates.append([convertSimilatiry(distance) for distance in distances_candidate])
		
	for candidate in features_candidates:
		distances_candidate = []
		for negative in generalDescriptor.negative_obj_model_features:
			dist = detSimilarity(candidate, negative)
			distances_candidate.append(dist)	# Lista das distancias em relacao as features negativas

		generalDescriptor.negative_distances_candidates.append(distances_candidate)	# Lista das distancias para cada candidato
		generalDescriptor.negative_similarity_candidates.append([convertSimilatiry(distance) for distance in distances_candidate])
	
	# Calculo das distancias e similaridades para a BB do candidato do Tracker
	for positive in generalDescriptor.positive_obj_model_features:
		dist = detSimilarity(generalDescriptor.tracker_features[LAST_ADDED], positive)
		generalDescriptor.positive_distances_tracker_candidate.append(dist)	# Lista das distancias em relacao as features positivas
		generalDescriptor.positive_similarity_tracker_candidate.append(convertSimilatiry(dist))

	for negative in generalDescriptor.negative_obj_model_features:
		dist = detSimilarity(generalDescriptor.tracker_features[LAST_ADDED], negative)
		generalDescriptor.negative_distances_tracker_candidate.append(dist)	# Lista das distancias em relacao as features negativas
		generalDescriptor.negative_similarity_tracker_candidate.append(convertSimilatiry(dist))

	#return bb_list_negativo, is_neg_empty, bb_list_positivo, is_pos_empty, bb_list_candidate, is_candidates_empty, bb_single_element_tracker, is_bb_tracker_empty

def TLD_parte_2(generated, imgs_pil):
	#retorno_frame = c_int()
	

	#print('general descriptor: ', generalDescriptor.positive_similarity_candidates)
	sim_pos_cand	= (c_float * len(generalDescriptor.positive_similarity_candidates))	(*generalDescriptor.positive_similarity_candidates)
	sim_neg_cand	= (c_float * len(generalDescriptor.negative_similarity_candidates))	(*generalDescriptor.negative_similarity_candidates)
	sim_pos_tracker = (c_float * len(generalDescriptor.positive_similarity_tracker_candidate)) (*generalDescriptor.positive_similarity_tracker_candidate)
	sim_neg_tracker = (c_float * len(generalDescriptor.negative_similarity_tracker_candidate)) (*generalDescriptor.negative_similarity_tracker_candidate)

	size_good_windows	  = c_int(0) 	# tamanho do vetor array good windows
	size_good_windows_hull = c_int(0) 	# tamanho do vetor array good_windows_hull (que e sempre 4)

	array_good_windows	  = [-1] * ARRAY_SIZE
	array_good_windows_hull = [-1] * ARRAY_SIZE

	array_good_windows	  = (c_float * ARRAY_SIZE) (*array_good_windows)
	array_good_windows_hull = (c_float * ARRAY_SIZE) (*array_good_windows_hull)

	shared_library.TLD_function_2(sim_pos_cand, sim_neg_cand,
								  sim_pos_tracker, sim_neg_tracker,
								  array_good_windows, byref(size_good_windows),
								  array_good_windows_hull, byref(array_good_windows_hull))

	bb_list_good_window, is_good_window_empty = read_data(array_good_windows, size_good_windows.value, frame)
	bb_good_windows_hull, is_good_hull_empty = read_data(array_good_windows_hull, size_good_windows_hull.value, frame)

	posicao = frame - 1

	addModel(generated, bb_list_good_window,  generalDescriptor.good_windows_bb,		generalDescriptor.good_windows_features, imgs_pil[posicao])
	addModel(generated, bb_good_windows_hull, generalDescriptor.good_windows_hull_bb,  generalDescriptor.good_windows_hull_features, imgs_pil[posicao])
	
	#return bb_list_good_window, is_good_window_empty, bb_good_windows_hull, is_good_hull_empty

def getOpts(opts):
	print("config opts...")
	opts['numScale'] = 3
	opts['scaleStep'] = 1.0375
	opts['scalePenalty'] = 0.9745
	opts['scaleLr'] = 0.59
	opts['responseUp'] = 16
	opts['windowing'] = 'cosine'
	opts['wInfluence'] = 0.176
	opts['exemplarSize'] = 127
	opts['instanceSize'] = 255
	opts['scoreSize'] = 17
	opts['totalStride'] = 8
	opts['contextAmount'] = 0.5
	opts['trainWeightDecay'] = 5e-04
	opts['stddev'] = 0.03
	opts['subMean'] = False
	opts['minimumSize'] = 87
	opts['video'] = 'vot15_bag'
	opts['modelPath'] = './models/'
	opts['modelName'] = opts['modelPath']+"model_tf.ckpt"
	opts['summaryFile'] = './data_track/'+opts['video']+'_20170518'

	return opts

def getAxisAlignedBB(region):
	region = np.array(region)
	nv = region.size
	assert (nv == 8 or nv == 4)

	if nv == 8:
		xs = region[0 : : 2]
		ys = region[1 : : 2]
		cx = np.mean(xs)
		cy = np.mean(ys)
		x1 = min(xs)
		x2 = max(xs)
		y1 = min(ys)
		y2 = max(ys)
		A1 = np.linalg.norm(np.array(region[0:2])-np.array(region[2:4]))*np.linalg.norm(np.array(region[2:4])-np.array(region[4:6]))
		A2 = (x2-x1)*(y2-y1)
		s = np.sqrt(A1/A2)
		w = s*(x2-x1)+1
		h = s*(y2-y1)+1
	else:
		x = region[0]
		y = region[1]
		w = region[2]
		h = region[3]
		cx = x+w/2
		cy = y+h/2

	return cx-1, cy-1, w, h

def frameGenerator(vpath):
	imgs = []
	imgFiles = [imgFile for imgFile in glob.glob(os.path.join(vpath, "*.jpg"))]
	for imgFile in imgFiles:
		if imgFile.find('00000000.jpg') >= 0:
			imgFiles.remove(imgFile)

	imgFiles.sort()

	for imgFile in imgFiles:
		# imgs.append(mpimg.imread(imgFile).astype(np.float32))
		# imgs.append(np.array(Image.open(imgFile)).astype(np.float32))
		img = cv2.imread(imgFile).astype(np.float32)
		imgs.append(img)

	return imgs

def loadVideoInfo(basePath, video):
	videoPath = os.path.join(basePath, video, 'imgs')
	groundTruthFile = os.path.join(basePath, video, 'groundtruth.txt')

	groundTruth = open(groundTruthFile, 'r')
	reader = groundTruth.readline()
	region = [float(i) for i in reader.strip().split(",")]
	cx, cy, w, h = getAxisAlignedBB(region)
	pos = [cy, cx]
	targetSz = [h, w]

	imgs = frameGenerator(videoPath)

	return imgs, np.array(pos), np.array(targetSz)

def getSubWinTracking(img, pos, modelSz, originalSz, avgChans):
	if originalSz is None:
		originalSz = modelSz

	sz = originalSz
	im_sz = img.shape
	# make sure the size is not too small
	assert min(im_sz[:2]) > 2, "the size is too small"
	c = (np.array(sz) + 1) / 2
	# check out-of-bounds coordinates, and set them to black
	context_xmin = round(pos[1] - c[1])
	context_xmax = context_xmin + sz[1] - 1
	context_ymin = round(pos[0] - c[0])
	context_ymax = context_ymin + sz[0] - 1
	left_pad = max(0, int(-context_xmin))
	top_pad = max(0, int(-context_ymin))
	right_pad = max(0, int(context_xmax - im_sz[1] + 1))
	bottom_pad = max(0, int(context_ymax - im_sz[0] + 1))
	context_xmin = int(context_xmin + left_pad)
	context_xmax = int(context_xmax + left_pad)
	context_ymin = int(context_ymin + top_pad)
	context_ymax = int(context_ymax + top_pad)
	if top_pad or left_pad or bottom_pad or right_pad:
		r = np.pad(img[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
				   constant_values=avgChans[0])
		g = np.pad(img[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
				   constant_values=avgChans[1])
		b = np.pad(img[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
				   constant_values=avgChans[2])
		r = np.expand_dims(r, 2)
		g = np.expand_dims(g, 2)
		b = np.expand_dims(b, 2)
		img = np.concatenate((r, g, b ), axis=2)
	im_patch_original = img[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1, :]
	if not np.array_equal(modelSz, originalSz):
		im_patch = cv2.resize(im_patch_original, modelSz)
	else:
		im_patch = im_patch_original

	return im_patch, im_patch_original

def makeScalePyramid(im, targetPosition, in_side_scaled, out_side, avgChans, stats, p):
	"""
	computes a pyramid of re-scaled copies of the target (centered on TARGETPOSITION)
	and resizes them to OUT_SIDE. If crops exceed image boundaries they are padded with AVGCHANS.

	"""
	in_side_scaled = np.round(in_side_scaled)
	max_target_side = int(round(in_side_scaled[-1]))
	min_target_side = int(round(in_side_scaled[0]))
	beta = out_side / float(min_target_side)
	# size_in_search_area = beta * size_in_image
	# e.g. out_side = beta * min_target_side
	search_side = int(round(beta * max_target_side))
	search_region, _ = getSubWinTracking(im, targetPosition, (search_side, search_side),
											  (max_target_side, max_target_side), avgChans)
	if p['subMean']:
		pass
	assert round(beta * min_target_side) == int(out_side)
	tmp_list = []
	tmp_pos = ((search_side - 1) / 2., (search_side - 1) / 2.)
	for s in range(p['numScale']):
		target_side = round(beta * in_side_scaled[s])
		tmp_region, _ = getSubWinTracking(search_region, tmp_pos, (out_side, out_side), (target_side, target_side),
											   avgChans)
		tmp_list.append(tmp_region)
	pyramid = np.stack(tmp_list)
	return pyramid

def trackerEval(score, sx, targetPosition, window, opts):
	# responseMaps = np.transpose(score[:, :, :, 0], [1, 2, 0])
	responseMaps = score[:, :, :, 0]
	upsz = opts['scoreSize']*opts['responseUp']
	# responseMapsUp = np.zeros([opts['scoreSize']*opts['responseUp'], opts['scoreSize']*opts['responseUp'], opts['numScale']])
	responseMapsUP = []

	if opts['numScale'] > 1:
		currentScaleID = int(opts['numScale']/2)
		bestScale = currentScaleID
		bestPeak = -float('Inf')
		for s in range(opts['numScale']):
			if opts['responseUp'] > 1:
				responseMapsUP.append(cv2.resize(responseMaps[s, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC))
			else:
				responseMapsUP.append(responseMaps[s, :, :])
			thisResponse = responseMapsUP[-1]
			if s != currentScaleID:
				thisResponse = thisResponse*opts['scalePenalty']
			thisPeak = np.max(thisResponse)
			if thisPeak > bestPeak:
				bestPeak = thisPeak
				bestScale = s
		responseMap = responseMapsUP[bestScale]
	else:
		responseMap = cv2.resize(responseMaps[0, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC)
		bestScale = 0
	responseMap = responseMap - np.min(responseMap)
	responseMap = responseMap/np.sum(responseMap)
	responseMap = (1-opts['wInfluence'])*responseMap+opts['wInfluence']*window
	rMax, cMax = np.unravel_index(responseMap.argmax(), responseMap.shape)
	pCorr = np.array((rMax, cMax))
	dispInstanceFinal = pCorr-int(upsz/2)
	dispInstanceInput = dispInstanceFinal*opts['totalStride']/opts['responseUp']
	dispInstanceFrame = dispInstanceInput*sx/opts['instanceSize']
	newTargetPosition = targetPosition+dispInstanceFrame
	# print(bestScale)

	return newTargetPosition, bestScale

'''----------------------------------------main-----------------------------------------------------'''
def main(_):
	
	print('run tracker...')
	opts = configParams()
	opts = getOpts(opts)
	#add
	minimumSiameseNetPlaceHolder = tf.placeholder(tf.float32, [1, opts['minimumSize'], opts['minimumSize'], 3])
	exemplarOp = tf.placeholder(tf.float32, [1, opts['exemplarSize'], opts['exemplarSize'], 3])
	instanceOp = tf.placeholder(tf.float32, [opts['numScale'], opts['instanceSize'], opts['instanceSize'], 3])
	exemplarOpBak = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['exemplarSize'], opts['exemplarSize'], 3])
	instanceOpBak = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['instanceSize'], opts['instanceSize'], 3])
	isTrainingOp = tf.convert_to_tensor(False, dtype='bool', name='is_training')
	sn = SiameseNet()
	scoreOpBak = sn.buildTrainNetwork(exemplarOpBak, instanceOpBak, opts, isTraining=False)
	saver = tf.train.Saver()
	writer = tf.summary.FileWriter(opts['summaryFile'])
	sess = tf.Session()
	saver.restore(sess, opts['modelName'])
	zFeatOp = sn.buildExemplarSubNetwork(exemplarOp, opts, isTrainingOp)
	zMinimumPreTrained =sn.buildExemplarSubNetwork(minimumSiameseNetPlaceHolder,opts,isTrainingOp)
	generated = Generation(opts,sn)
	#generated.getDescriptor(coordenadasDaImagem,Image.open('download.jpeg'))
	imgs, targetPosition, targetSize = loadVideoInfo(opts['seq_base_path'], opts['video'])
	nImgs = len(imgs)
	imgs_pil =  [Image.fromarray(np.uint8(img)) for img in imgs]

		
	im = imgs[POSICAO_PRIMEIRO_FRAME]
	if(im.shape[-1] == 1):
		tmp = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.float32)
		tmp[:, :, 0] = tmp[:, :, 1] = tmp[:, :, 2] = np.squeeze(im)
		im = tmp

	avgChans = np.mean(im, axis=(0, 1))# [np.mean(np.mean(img[:, :, 0])), np.mean(np.mean(img[:, :, 1])), np.mean(np.mean(img[:, :, 2]))]
	wcz = targetSize[1]+opts['contextAmount']*np.sum(targetSize)
	hcz = targetSize[0]+opts['contextAmount']*np.sum(targetSize)
	sz = np.sqrt(wcz*hcz)
	scalez = opts['exemplarSize']/sz

	zCrop, _ = getSubWinTracking(im, targetPosition, (opts['exemplarSize'], opts['exemplarSize']), (np.around(sz), np.around(sz)), avgChans)

	if opts['subMean']:
		pass

	dSearch = (opts['instanceSize']-opts['exemplarSize'])/2
	pad = dSearch/scalez
	sx = sz+2*pad

	minSx = 0.2*sx
	maxSx = 5.0*sx
	winSz = opts['scoreSize']*opts['responseUp']
	if opts['windowing'] == 'cosine':

		hann = np.hanning(winSz).reshape(winSz, 1)
		window = hann.dot(hann.T)
	elif opts['windowing'] == 'uniform':
		window = np.ones((winSz, winSz), dtype=np.float32)

	window = window/np.sum(window)
	scales = np.array([opts['scaleStep'] ** i for i in range(int(np.ceil(opts['numScale']/2.0)-opts['numScale']), int(np.floor(opts['numScale']/2.0)+1))])
	zCrop2 = np.array(zCrop)
	zCrop = np.expand_dims(zCrop, axis=0)
	zCropMinimum = Image.fromarray(zCrop2,'RGB')
	zCropMinimum = zCropMinimum.resize([ATOMIC_SIZE,ATOMIC_SIZE])
	zCropMinimum = np.array(zCropMinimum)
	zCropMinimum = np.expand_dims(zCropMinimum, axis=0)
	zFeat = sess.run(zFeatOp, feed_dict={exemplarOp: zCrop})
	zMinimumFeatures = sess.run(zMinimumPreTrained, feed_dict={minimumSiameseNetPlaceHolder: zCropMinimum})
	zMinimumFeatures = np.reshape(zMinimumFeatures,[DIM_DESCRIPTOR,ONE_DIMENSION])
	zFeat = np.transpose(zFeat, [1, 2, 3, 0])
	zFeatConstantOp = tf.constant(zFeat, dtype=tf.float32)
	scoreOp = sn.buildInferenceNetwork(instanceOp, zFeatConstantOp, opts, isTrainingOp)
	writer.add_graph(sess.graph)
	resPath = os.path.join(opts['seq_base_path'], opts['video'], 'res')
	bBoxes = np.zeros([nImgs, 4])
	tic = time.time()
	for frame in range(POSICAO_PRIMEIRO_FRAME, nImgs):
		print('')
		print(('Estamos no frame ' + str(frame+1)).center(80,'*'))
		if frame > POSICAO_PRIMEIRO_FRAME:
			im = imgs[frame]
			if(im.shape[-1] == 1):
				tmp = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.float32)
				tmp[:, :, 0] = tmp[:, :, 1] = tmp[:, :, 2] = np.squeeze(im)
				im = tmp
			scaledInstance = sx * scales
			scaledTarget = np.array([targetSize * scale for scale in scales])
			xCrops = makeScalePyramid(im, targetPosition, scaledInstance, opts['instanceSize'], avgChans, None, opts)
			# sio.savemat('pyra.mat', {'xCrops': xCrops})
			score = sess.run(scoreOp, feed_dict={instanceOp: xCrops})
			sio.savemat('score.mat', {'score': score})
			newTargetPosition, newScale = trackerEval(score, round(sx), targetPosition, window, opts)
			targetPosition = newTargetPosition
			sx = max(minSx, min(maxSx, (1-opts['scaleLr'])*sx+opts['scaleLr']*scaledInstance[newScale]))
			targetSize = (1-opts['scaleLr'])*targetSize+opts['scaleLr']*scaledTarget[newScale]

			TLD_parte_1(generated, imgs_pil, frame+1)
			TLD_parte_2(generated, imgs_pil)

		else:
			init_TLD_in_siameseFC(generated, imgs_pil, frame+1)
			#pass
		rectPosition = targetPosition-targetSize/2.
		tl = tuple(np.round(rectPosition).astype(int)[::-1])
		br = tuple(np.round(rectPosition+targetSize).astype(int)[::-1])
		imDraw = im.astype(np.uint8)
		cv2.rectangle(imDraw, tl, br, (0, 255, 255), thickness=3)
		cv2.imshow("tracking", imDraw)
		cv2.waitKey(1)

	print(time.time()-tic)
	return

#mal feito ou incompleto(?)
def addModel(generated, bb_list, bb_acumulated_atribute, feature_acumulated_atribute, image):
	for bb in bb_list:
		currentFeature = generated.getDescriptor(bb, image)
		#print('\n\n\n\n##################  ',currentFeature)
		bb_acumulated_atribute.append(bb)
		feature_acumulated_atribute.append(currentFeature)

######################## Core da interface TLD ########################

# Criar a lista de imagens que serao processadas no rastreamento



'''	
for frame, posicao in zip(range(SEGUNDO_FRAME,ULTIMO_FRAME),range(POSICAO_SEGUNDO_FRAME,ULTIMO_FRAME-1) ):
	bb_list_negativo, _ , bb_list_positivo, _ , bb_list_candidato, _, bb_single_element_tracker, _ = TLD_parte_1(frame)

	addModel(generated, bb_list_negativo, generalDescriptor.negative_obj_model_bb, generalDescriptor.negative_obj_model_features, imgs_pil[posicao])
	addModel(generated, bb_list_positivo, generalDescriptor.positive_obj_model_bb, generalDescriptor.positive_obj_model_features, imgs_pil[posicao])
	addModel(generated, bb_single_element_tracker, generalDescriptor.tracker_bb, generalDescriptor.tracker_features, imgs_pil[posicao])
	
	list_feature = []
	list_bb 	 = []
	addModel(generated, bb_list_candidate, list_bb, list_feature, imgs_pil[posicao])
	generalDescriptor.setCandidates(list_bb,list_feature, frame)

	# Calculo das distancias e similaridades para os candidatos	
	positive_distances_candidates  = []
	negative_distances_candidates  = []
	positive_similarity_candidates = []
	negative_similarity_candidates = []

	positive_distances_tracker_candidate  = []
	negative_distances_tracker_candidate  = []
	positive_similarity_tracker_candidate = []
	negative_similarity_tracker_candidate = []

	features_candidates = generalDescriptor.getCandidates(currentFrameNumber)

	for candidate in features_candidates:
		distances_candidate = []
		for positive in generalDescriptor.positive_obj_model_features:
			dist = detSimilarity(candidate, positive)
			distances_candidate.append(dist)	# Lista das distancias em relacao as features positivas
			
		positive_distances_candidates.append(distances_candidate)	# Lista das distancias para cada candidato
		positive_similarity_candidates.append([convertSimilatiry(distance) for distance in distances_candidate])
		
	for candidate in features_candidates:
		distances_candidate = []
		for negative in generalDescriptor.negative_obj_model_features:
			dist = detSimilarity(candidate, negative)
			distances_candidate.append(dist)	# Lista das distancias em relacao as features negativas

		negative_distances_candidates.append(distances_candidate)	# Lista das distancias para cada candidato
		negative_similarity_candidates.append([convertSimilatiry(distance) for distance in distances_candidate])
	
	# Calculo das distancias e similaridades para a BB do candidato do Tracker
	for positive in generalDescriptor.positive_obj_model_features:
		dist = detSimilarity(generalDescriptor.tracker_feature[LAST_ADDED], positive)
		positive_distances_tracker_candidate.append(dist)	# Lista das distancias em relacao as features positivas
		positive_similarity_tracker_candidate.append(convertSimilatiry(dist))

	for negative in generalDescriptor.negative_obj_model_features:
		dist = detSimilarity(generalDescriptor.tracker_feature[LAST_ADDED], negative)
		negative_distances_tracker_candidate.append(dist)	# Lista das distancias em relacao as features negativas
		negative_similarity_tracker_candidate.append(convertSimilatiry(dist))

	bb_list_good_window, _, bb_good_windows_hull, _ = TLD_parte_2()

	addModel(generated, bb_list_good_window,  generalDescriptor.good_windows_bb,		generalDescriptor.good_windows_features,	   0image)
	addModel(generated, bb_good_windows_hull, generalDescriptor.good_windows_hull_bb,  generalDescriptor.good_windows_hull_features, 0image)
'''


######################  ~Core da interface TLD ########################

if __name__=='__main__':
	tf.app.run()
