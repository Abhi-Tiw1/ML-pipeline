#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils file for functions of ml_pipeline
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold,StratifiedKFold
#from utils.ml_utils import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import os #balanced_
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import precision_score,recall_score,cohen_kappa_score, matthews_corrcoef
import pymrmr




def make_out_vec(df_ind):
	"""returns output vector with mean and standard deviaiton for a given configuration """		
	out_col=df_ind.columns.values.tolist()
	oc_mean=add_to_list(out_col,'_mn')
	oc_std=add_to_list(out_col,'_std')
	oc_fin=oc_mean+oc_std
	
	out_mean=np.around(np.mean(df_ind),decimals=4)
	out_std=np.around(np.std(df_ind),decimals=4)
	out_all=np.hstack(((np.array(out_mean),(np.array(out_std)))))

	return out_all,oc_fin
	
def save_df_results(arr_,col,outpath_fin,fnm):
	
	final=pd.DataFrame(arr_,columns=col)
	final.to_csv(outpath_fin+fnm,index=None)

def save_perfold_res(arr_,col_,ft_nms,out_dir):
	"""Per fold results saved by calling this function """
	df_ou=pd.DataFrame(arr_,columns=col_)
	top_feats,feat_freq=np.unique(ft_nms,return_counts=True)
	top_feats=top_feats[np.argsort(feat_freq)]
	feat_freq=np.sort(feat_freq)
	df_feat_ana=pd.DataFrame(np.vstack((top_feats,feat_freq)).T,columns=['Features','Freq']).iloc[::-1]
	
	fnm=out_dir+'out_perf_fold.csv'
	fnm_ft=out_dir+'out_ft_ana.csv'
	df_ou.to_csv(fnm,index=None)
	df_feat_ana.to_csv(fnm_ft,index=None)
	out_vec=np.mean(df_ou,axis=0)
	return out_vec


def get_sorted_counts(arr):
	arr,counts=np.unique(arr,return_counts=True)
	ind=np.argsort(counts)
	counts=counts[ind]
	arr=arr[ind]
	return arr,counts


def mRMR_sel(X_tr,X_te,y_tr,k,feat_name):

	X_tr,X_te,feat_name=select_fs_alg('anova', X_tr, X_te, y_tr, 500, feat_name)
	if X_tr.shape[1]<k:
		X_t=X_tr
		mr_feat=feat_name
		X_te=X_te
		return X_t,X_te,mr_feat


	data=np.concatenate([np.expand_dims(y_tr,1),X_tr],axis=1)
	fin_name=np.hstack((np.array('tar'),feat_name))
	df=pd.DataFrame(data,columns=fin_name)
	df_te=pd.DataFrame(X_te,columns=feat_name)
	mr_feat=pymrmr.mRMR(df, 'MIQ', k)
	X_t=np.array(df[mr_feat])
	X_te = np.array(df_te[mr_feat])
	
	return X_t,X_te,mr_feat
	

def ANOVA_sel(X_tr,X_te,y_tr,k,feat_name):
	
	Anova_sel= SelectKBest(f_classif, k)
	if X_tr.shape[1]<k:
		X_t=X_tr
		mask=feat_name
		X_te=X_te
		return X_t,X_te,mask
	
	X_t= Anova_sel.fit_transform(X_tr, y_tr)
	#ind=np.where(Anova_sel.pvalues_<0.05)
	ind=Anova_sel.get_support()
	mask=feat_name[ind]
	#X= Anova_sel.transform(feats)
	X_te = Anova_sel.transform(X_te)
	return X_t,X_te,mask



def RFE_fs(clf_rfe,X_tr,X_te,y_tr,k,feat_name,step=10):
	
	
	X_tr,X_te,feat_name=select_fs_alg('anova', X_tr, X_te, y_tr, 500, feat_name)
	RFE_sel = RFE(clf_rfe, k, step=step)
	if X_tr.shape[1]<k:
		X_t=X_tr
		mask=feat_name
		X_te=X_te
		return X_t,X_te,feat_name
	
	X_t = RFE_sel.fit_transform(X_tr,y_tr) #
	X_te = RFE_sel.transform(X_te)
	mask=RFE_sel.support_
	return X_t,X_te,feat_name[mask]


def select_fs_alg(fs_alg, X_tr, X_te, y_tr, nof, feats):
	
	"""
	Does the feature selection using 3 different algorithms
	Input:  
	X_tr: training set feature matrix  
	y_tr: training set labels  
 	X_te: test set feature matrix  
 	nof: number of features to be selected  
 	feats: input set feature names  
 	fs_alg- Input key for selecting feature selection method. Options are:
 	  - 'rfe': recurcive feature elimination
 	  - 'mrmr': mrmr feature selection
 	  - 'anova': anova based feature selection
 	Outputs:  
 	X_tr_fs: training matrix after feature selection  
 	X_trans : test matrix after feature selection
 	mask: feature names of selected features	
	"""
	
	if fs_alg =='anova':
		X_tr_fs,X_trans,mask=ANOVA_sel(X_tr,X_te,y_tr,nof,feats)
	elif fs_alg == 'rfe':
		clf_rfe=ExtraTreesClassifier(n_estimators=100)
		X_tr_fs,X_trans,mask=RFE_fs(clf_rfe,X_tr,X_te,y_tr,nof,feats,step=10)
	elif fs_alg== 'mrmr':
		X_tr_fs,X_trans,mask=mRMR_sel(X_tr,X_te,y_tr,nof,feats)


	return X_tr_fs, X_trans, mask
	
def select_clf(clfr):

	"""
	Function to call classifier key (used in get_cv_out function)  
	Input:  
	clfr: Input key for classfier selection. Options are:  
		  - svm_rbf: SVM with rbf kernal  
		  - svm_lnr: SVM with linear kernal  
		  - lr: logistic regression  
		  - rf20: random forest with 20 treees  
		  - knn10: k-nearest neighbors with 10 neighbors  
		  
	Output:  
	clf: selected classifier	  
	"""
	
	if clfr=='svm_rbf':
		clf=svm.SVC(kernel='rbf',C=1,gamma='auto',class_weight='balanced')
	elif clfr=='svm_lnr':
		clf=svm.SVC(kernel='linear',class_weight='balanced')
	elif clfr=='lr':
		clf= LogisticRegression(random_state=0)
	elif clfr =='rf20':
		clf=RandomForestClassifier(n_estimators=20)
	elif clfr=='knn10':
		clf= KNeighborsClassifier(n_neighbors=10)
		
	return clf

def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None,
							adjusted=False):
   
	C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
	with np.errstate(divide='ignore', invalid='ignore'):
		per_class = np.diag(C) / C.sum(axis=1)
	if np.any(np.isnan(per_class)):
		warnings.warn('y_pred contains classes not in y_true')
		per_class = per_class[~np.isnan(per_class)]
	score = np.mean(per_class)
	if adjusted:
		n_classes = len(per_class)
		chance = 1 / n_classes
		score -= chance
		score /= 1 - chance
	return score

def get_metrics(y_tru, y_pred,no_metrics=6):
	"""
	Funciton to get the output classification metrics (called in get_cv_out)  
	Inputs:  
	y_tru : the true output labels  
	y_pred: rhe predicted output labels  
	no_metrics: default 6, the number of output metrics   
	
	Outputs:  
	vec: returns vector with following order:   
		- balanced accuracy,  
		- f1-score,  
		- matthews correlation coefficient,  
		- precision,  
		- recall,  
		- cohen's kappa score,   
	
	"""

	vec=np.empty((no_metrics))
	vec[0]=balanced_accuracy_score(y_tru,y_pred)  #balanced_
	vec[1]=f1_score(y_tru,y_pred)
	vec[2]=matthews_corrcoef(y_tru,y_pred)
	vec[3]=precision_score(y_tru,y_pred)
	vec[4]=recall_score(y_tru,y_pred)
	vec[5]=cohen_kappa_score(y_tru,y_pred)
	
	return vec
	
def get_cv_out(X,y,fs_alg,nof,feats,clfrs,samp_type,rseed):
	"""
	Performs an 5 fold cross validation on the input data along with resampling (for imbalabced data) ,feature selection,
	and with different classifier and reruens the output performance metrics for each fold along with features selected.  
	Inputs:  
	X - feature matrix for full dataset  
	y - labels for full dataset  
	fs_alg - feature selection algorithm  
	nof - number of features to select  
	feats:  input feature names  
	clfrs: list of classifiers to be used  
	samp_typr: sampling strategy (for imbalabced data)  
	rseed: random seed for cross validation
	
	Outputs:  
	f_names: name of all features selected in each fold  
	out_fold: performance metrics for each fold and classifier along with random voting outputs
	
	"""
	
	np.random.seed(rseed)
	no_folds=5
	outer_cv=KFold(n_splits=no_folds, shuffle=True)
	no_clf=len(clfrs)
	#array for accuracy and f1 per fold per clasifier
	no_metrics=6  # acc, f1, mcc, prc, rec, coh, random_ba
	
	f_names=np.empty((0))
	out_fold=np.zeros((0,(no_clf+1)*no_metrics))


	for train_index, test_index in outer_cv.split(X,y):	
	
		
		X_tr, X_te = X[train_index], X[test_index]
		y_tr, y_te = y[train_index], y[test_index]
		
		
		X_tr, y_tr =get_sampled_data(X_tr,y_tr,rseed,samp_type)
		scaler=preprocessing.StandardScaler().fit(X_tr)
		X_tr=scaler.transform(X_tr)
		X_te=scaler.transform(X_te)
		
		X_tr_fs,X_trans,mask=select_fs_alg(fs_alg, X_tr, X_te, y_tr, nof, feats)

		f_names=np.hstack((f_names,mask))
		
		out_arr=np.zeros((no_clf,no_metrics))
		rv_arr=np.zeros((no_metrics))
		
		for i_clf,clfr in enumerate(clfrs):
			clf= select_clf(clfr)
			clf.fit(X_tr_fs,y_tr)
			y_pre=clf.predict(X_trans)
			#confusion matrix per clasifier/ per fold
			y_rv=(np.random.rand(len(y_pre))>0.5)*1
		
			#confusion matrix per clasifier/ per fold
			out_arr[i_clf,:]=get_metrics(y_te,y_pre,no_metrics)
		
		rv_arr=get_metrics(y_te,y_rv,no_metrics)
		   
		out_vec=np.hstack((out_arr.flatten(),rv_arr.flatten()))
		out_fold=np.vstack((out_fold,out_vec))
		
		
	return f_names, out_fold

def get_out_col_names(c,m):
	out=[]
	out_rv=[]
	for cl in c:
		for ml in m:
			out.append(cl+'_'+ml)
		
	for ml in m:
		out_rv.append(ml+'_rv')
	return out+out_rv

def get_sampled_data(Xsm, ysm, seed,type='over'):
	"""
	Function samples the imbalabced dataset to make it balanced. To be called for the training set data 
	Inputs: 
	Xsm - features  to be sampled
	ysm - corrsponding labels of feature
	seed - random seed for sampling
	type - type of sampling, options are
		   - 'under' : random undersampling
		   - 'over' : random oversampling
		   - 'under_nm3' : undersampling using near miss method
		   - 'over_smote' : oversampling using smote methselectionod
		   
	Outputs: 
	- X_rs, y_rs: resampled features and labels
	"""
	np.random.seed(seed)
	if type=='under':
		rus = RandomUnderSampler(random_state=seed)
		X_rs, y_rs= rus.fit_resample(Xsm, ysm)
	elif type=='over':
		ros = RandomOverSampler(random_state=seed)
		X_rs, y_rs= ros.fit_resample(Xsm, ysm)
	elif type=='under_nm3':
		nm3= NearMiss(version=3)
		X_rs, y_rs= nm3.fit_resample(Xsm, ysm)
	elif type=='over_smote':
		X_rs, y_rs= SMOTE().fit_resample(Xsm, ysm)
	else:
		X_rs, y_rs= Xsm, ysm
	
	return X_rs,y_rs
	





