import numpy as np

mlp_res = np.load('res_e1_mlp.npy')
mlp_errs = np.load('estim_err_mlp.npy')[:,:,:-1]

sis_res = np.load('res_e1_sis.npy')
sis_errs = np.load('estim_err_sis.npy')[:,:,:-4]

# temp
cdis_res = np.load('res_e1_sis.npy')
cdis_errs = np.load('estim_err_sis.npy')[:,:,:-4]
# temp
ddis_res = np.load('res_e1_sis.npy')
ddis_errs = np.load('estim_err_sis.npy')[:,:,:-4]


# res
mlp_raw = mlp_res[:,:,0]
sis_raw = sis_res[:,:,:4]
cdis_raw = cdis_res[:,:,:4]
ddis_raw = ddis_res[:,:,:4]

mlp_methods = mlp_res[:,:,1:]
sis_methods = sis_res[:,:,4:]
cdis_methods = cdis_res[:,:,4:]
ddis_methods = ddis_res[:,:,4:]

mlp_raw = mlp_raw.reshape((10,9,1,499,1))

res_non_mlp_raw = np.concatenate((sis_raw, cdis_raw, ddis_raw), axis=1)
res_non_mlp_methods = np.concatenate((sis_methods, cdis_methods, ddis_methods), axis=1)

res_all = np.concatenate((mlp_raw, res_non_mlp_raw, mlp_methods, res_non_mlp_methods), axis=2)
print(res_all.shape)

np.save('res_e1_all', res_all)

# errs

errs_non_mlp = np.concatenate((sis_errs, cdis_errs, ddis_errs), axis=1)
errs_all = np.concatenate((mlp_errs, errs_non_mlp), axis=2)

print(errs_all.shape)

np.save('errs_e1_all', errs_all)