import os

CAMINHO_VOT2015 = '/home/swhants/Documentos/vot2015/'
listVideos = [ i for i in os.listdir(CAMINHO_VOT2015) if not i.startswith('_')]

for video in listVideos:
	os.system('python3.6 tracker.py -v'+ video)