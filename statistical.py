import numpy as np

res = np.load('results/res_e1_all.npy')
print(res.shape)

# shape: reps x streams x methods x chunks-1 x metric (bac)
# methods : pierwsze 5 to same klasyfikatory - kolejno MLP, GNB, KNN, SVM, HTC
#           później: (base_clfs(5 szt), criteria(min, max) x borders(od 1% do 50%) x estimators(mean, prev, dsca))
#       
# najlepsze wartości parametrów trzeba wyprać na podstawie e2

res_raw_clfs = res[:,:,:5]
print(res_raw_clfs.shape)

res_methods = res[:,:,5:]
res_methods = res_methods.reshape(10,9,5,2,10,3,499)
print(res_methods.shape) # reps x streams x base clfs x criterion x borders x estimators x chunks