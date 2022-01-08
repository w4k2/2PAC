import numpy as np

mlp_res = np.load('results/res_e1_mlp.npy')
mlp_errs = np.load('results/estim_err_mlp.npy')[:,:,:-1]

sis_res = np.load('results/res_e1_sis.npy')
sis_errs = np.load('results/estim_err_sis.npy')[:,:,:-4]

cdis_res = np.load('results/res_e1_cdis.npy')
cdis_errs = np.load('results/estim_err_cdis.npy')[:,:,:-4]

# ddis (dramat) gnb, knn, svc, htc

ddis_res_gnb = np.load('results/res_e1_gnb.npy')[:,-3:]
# temp
ddis_res_htc = np.load('results/res_e1_gnb.npy')[:,-3:]
ddis_res_svc = np.load('results/res_e1_gnb.npy')[:,-3:]
ddis_res_knn = np.load('results/res_e1_gnb.npy')[:,-3:]

ddis_raw_gnb = ddis_res_gnb[:,:,0]
ddis_methods_gnb = ddis_res_gnb[:,:,1:]

ddis_raw_htc = ddis_res_htc[:,:,0]
ddis_methods_htc = ddis_res_htc[:,:,1:]

ddis_raw_svc = ddis_res_svc[:,:,0]
ddis_methods_svc = ddis_res_svc[:,:,1:]

ddis_raw_knn = ddis_res_knn[:,:,0]
ddis_methods_knn = ddis_res_knn[:,:,1:]


ddis_raw = np.concatenate((ddis_raw_gnb[:,:,np.newaxis], ddis_raw_knn[:,:,np.newaxis], ddis_raw_svc[:,:,np.newaxis], ddis_raw_htc[:,:,np.newaxis]), axis=2)
ddis_methods = np.concatenate((ddis_methods_gnb, ddis_methods_knn, ddis_methods_svc, ddis_methods_htc), axis=2)
print(ddis_methods.shape)

# ------------

ddis_errs_gnb = np.load('results/estim_err_gnb.npy')[:,-3:,:-1]
# temp
ddis_errs_htc = np.load('results/estim_err_gnb.npy')[:,-3:,:-1]
ddis_errs_svc = np.load('results/estim_err_gnb.npy')[:,-3:,:-1]
ddis_errs_knn = np.load('results/estim_err_gnb.npy')[:,-3:,:-1]

ddis_errs = np.concatenate((ddis_errs_gnb, ddis_errs_knn, ddis_errs_svc, ddis_errs_htc), axis=2)
print(ddis_errs.shape)

# res
mlp_raw = mlp_res[:,:,0]
sis_raw = sis_res[:,:,:4]
cdis_raw = cdis_res[:,:,:4]

mlp_methods = mlp_res[:,:,1:]
sis_methods = sis_res[:,:,4:]
cdis_methods = cdis_res[:,:,4:]

mlp_raw = mlp_raw.reshape((10,9,1,499,1))

res_non_mlp_raw = np.concatenate((sis_raw, cdis_raw, ddis_raw), axis=1)
res_non_mlp_methods = np.concatenate((sis_methods, cdis_methods, ddis_methods), axis=1)

res_all = np.concatenate((mlp_raw, res_non_mlp_raw, mlp_methods, res_non_mlp_methods), axis=2)
print(res_all.shape)

np.save('results/res_e1_all', res_all)

# errs

errs_non_mlp = np.concatenate((sis_errs, cdis_errs, ddis_errs), axis=1)
errs_all = np.concatenate((mlp_errs, errs_non_mlp), axis=2)

print(errs_all.shape)

np.save('results/errs_e1_all', errs_all)