from scipy import stats
import numpy as np
import config

weights = config.str_weights()
weigths_names = config.str_weights_names()
borders = config.borders()
print(borders)

base_clfs_names = config.base_clf_names()
pe = ["MEAN", "PREV", "DSCA"]

reps = 10
chunks = 500

alpha = .05

wn = np.array(weigths_names)

res = np.load('results/res_e1_all.npy')[:,:,:,:,0]
# reps, weights, methods, chunks
print(res.shape)

meta_res = res[:,:,len(base_clfs_names):].reshape((reps,len(weights),len(base_clfs_names),2,len(borders),3,chunks-1))
meta_res = meta_res[:,:,:,0]

print(meta_res.shape)
# reps, weights, clfs, thresholds, pe, chunks


res_selected = []

#sis -- threshold max i mean
res_selected.append(meta_res[:,:3,:,-1,0,:])

#cdis -- threshold 0.28 i prev
res_selected.append(meta_res[:,3:6,:,5,1,:])

#ddis -- threshold max i prev
res_selected.append(meta_res[:,6:,:,-1,2,:])

res_selected = np.array(res_selected).swapaxes(0,1).reshape(10,9,5,499)
print(res_selected.shape)

res_baseline = meta_res[:,:,:,0,0,:]
print(res_baseline.shape)

all_results = np.array([
    res_baseline[:,:,0], # mlp
    res_selected[:,:,0], # mlp-2pac
    res_baseline[:,:,1], # gnb
    res_selected[:,:,1], # gnb-2pac
    res_baseline[:,:,2], # knn
    res_selected[:,:,2], # knn-2pac
    res_baseline[:,:,3], # svm
    res_selected[:,:,3], # svm-2pac
    res_baseline[:,:,4], # htc
    res_selected[:,:,4], # htc-2pac
])

all_results = all_results.swapaxes(0,1)

print(all_results.shape) # reps, clfs, streams, chunks


methods = ['MLP', '2PAC-MLP', 
           'GNB', '2PAC-GNB',
           'KNN', '2PAC-KNN',
           'SVM', '2PAC-SVM',
           'HTC', '2PAC-HTC'
           ]

all_results = np.mean(all_results, axis=-1) # średnia w chunkach

mean_res = np.mean(all_results, axis=0) # methods, weights
std_res = np.std(all_results, axis=0)


rows = []
for w_id, w in enumerate(weigths_names):
    
    r_temp = all_results[:,:,w_id]
    print(r_temp.shape) # reps, methods
    
    t_stat = np.zeros((len(methods), len(methods)))
    p_val = np.zeros((len(methods), len(methods)))
    better = np.zeros((len(methods), len(methods))).astype(bool)

    for i in range(len(methods)):
        for j in range(len(methods)):
            t_stat[i,j], p_val[i,j] = stats.ttest_rel(r_temp[:,i], r_temp[:,j])
            better[i,j] = np.mean(r_temp[:,i]) > np.mean(r_temp[:,j])
            
    significant = p_val<alpha
    significantly_better = significant*better

    print(significantly_better)
        
    r = []
    r.append('W: %s' % (w))
    for m_id, m in enumerate(methods):
        r.append(np.round(mean_res[m_id, w_id],3))
    rows.append(r)   
    
    r = []
    r.append('')
    for m_id, m in enumerate(methods):
        better = np.argwhere(significantly_better[m_id]==True).flatten()
        if len(better)==9:
            r.append('all')           

        else:
            r.append(' '.join(map(str,better)))           
    rows.append(r)

from tabulate import tabulate
print(tabulate(rows, methods, tablefmt="latex"))
# print(tabulate(rows, methods, tablefmt="simple"))
    
    
