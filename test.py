import numpy as np

'''
lista1 = []
lista2 = []

lista2.append(lista1)

print(lista2)

for i in lista2:
	print(i)
'''

SIZE_DESCRIPTOR = 32

def getDescriptor():
	descriptor = []
	#TODO Estamos colocando apenas um place holder. A funcao depende da analise do tracker siameseFC no python
	for _ in range(SIZE_DESCRIPTOR):
		descriptor.append(float(np.random.randn()))
	
	return descriptor

'''
descriptor = getDescriptor()
print(len(descriptor))
print(descriptor)
'''

'''
descriptor = np.random.randn(1,32)
print(descriptor.shape)
print(descriptor)

a = np.asarray(descriptor)
lista = []
for _ in range(3):
	lista.append(a)

print(lista)

lista = np.asarray(lista)
print(lista)
'''

'''
a.append(np.asarray(descriptor))
a.append(np.asarray(descriptor))
print(a)
a = np.asarray(a)

print(a)
print(type(a))
'''

bb_list = []
bb_pos = []

for i in range(32):
	bb_pos.append(i)

	if(i%4==0 and i is not 0):
		bb_pos.append(i)
		bb_list.append(bb_pos)
		bb_pos = []
		print('')
		
	print('bb_list: '+str(bb_list))
print('bb_list: '+str(bb_list))