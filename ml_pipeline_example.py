
"""
Example code
Code implements the ml pipeline on the iris dataset to do binary calssification
"""

from sklearn import datasets
from ml_pipeline import *

def run_ml_pipeline(X, y, fs_alg, nof, feats, clfrs, samp_type, no_out_vec, no_iters=50):
	"""Calls get_cv_out for X number of iternation and saves the results """
	
	#code gets the average cv values and other things for all the different
	tot_metric=np.empty((0,no_out_vec))
	f_names_tot=np.empty((0))
	no_iters=50
	#output for a single fold --> repeated 50 times
	for rseed in range(0,no_iters,1):
		f_names,out_metric=get_cv_out(X,y,fs_alg,nof,np.array(feats),clfrs,samp_type,rseed)
		tot_metric=np.vstack((tot_metric,out_metric))
		f_names_tot=np.hstack((f_names_tot,f_names))
		
	return f_names_tot, tot_metric
	

iris = datasets.load_iris()
X = iris.data
y = iris.target
#making the problem binary
X=X[:100,:]
y=y[:100]
target_names = iris.target_names

#feature names
feats=np.array(['1','2','3','4'])
#sampling type
samp_type='none'
#number of features given as 2 out of 4
nof=2
#classifiers to use
clfrs=['svm_rbf', 'svm_lnr','rf20', 'knn10','lr'] 
# feature selection algorithm - recurcive feature elimination 
fs_alg='rfe'
#metric names for performance measurement
metrics_= ['bacc','f1','mcc','pre','rec','coh']
no_metrics=len(metrics_)
#output column names --> classifier + metric
out_col_perfold=get_out_col_names(clfrs,metrics_)

#final colum which also stores informaiton about balance of dataset
cols = out_col_perfold+['balance']

no_fin_cols=len(cols)
#open the design matrix
fin_arr_out=np.empty((0,no_fin_cols))

print('Balance is',np.round(np.mean(y),3))
print('Shape of arrays is ',X.shape,y.shape,'\n--------------')

no_out_vec=len(out_col_perfold)

f_names_tot, tot_metric= run_ml_pipeline(X,y,fs_alg,nof,feats,clfrs, samp_type, no_out_vec)
#saves the feature analysis for this feature selection aglo
outpath_fin='./ml_pipeline_out/'
if not os.path.exists(outpath_fin):
	os.makedirs(outpath_fin)
out_vec=save_perfold_res(tot_metric,out_col_perfold,f_names_tot,outpath_fin)
out_vec=np.round(out_vec,3)
out_vec=np.hstack((out_vec,np.round(np.mean(y),3)))
#results for a given pwl level and feat selection method
fin_arr_out=np.vstack((fin_arr_out,out_vec))
	
#saving for all pwls and given lobe
fnm_all='output_fin.csv'
save_df_results(fin_arr_out,cols,outpath_fin,fnm_all)	




