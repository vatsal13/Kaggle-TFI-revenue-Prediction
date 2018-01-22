#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Sukriti'

# Reference : https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import pandas as pd
from datetime import datetime
import sys


# Global Variables
train = ''
test = ''
Y = ''

# After amputation
def categorical2numerical(glob_train, glob_test):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)
	
	leCity = preprocessing.LabelEncoder()
	leCity.fit(loc_train['City'])
	loc_train['City'] = leCity.transform(loc_train['City'])
	loc_test['City'] = leCity.transform(loc_test['City'])

	leCityG = preprocessing.LabelEncoder()
	leCityG.fit(loc_train['City Group'])
	loc_train['City Group'] = leCityG.transform(loc_train['City Group'])
	loc_test['City Group'] = leCityG.transform(loc_test['City Group'])

	leType = preprocessing.LabelEncoder()
	leType.fit(loc_train['Type'])
	loc_train['Type'] = leType.transform(loc_train['Type'])
	loc_test['Type'] = leType.transform(loc_test['Type'])

	return {'train': loc_train ,'test': loc_test}


# Normalize - sqrt
def normalize_sqrt(glob_train, glob_test, glob_Y):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)
	loc_train = loc_train.apply(np.sqrt)
	loc_test = loc_test.apply(np.sqrt)
	loc_Y = loc_Y.apply(np.sqrt)

	return {'train': loc_train ,'test': loc_test, 'Y': loc_Y}


# Normalize - log
def normalize_log(glob_train, glob_test, glob_Y):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)
	loc_train = loc_train.apply(lambda x: np.log(x+1))
	loc_test = loc_test.apply(lambda x: np.log(x+1))
	loc_Y = loc_Y.apply(lambda x: np.log(x+1))

	return {'train': loc_train ,'test': loc_test, 'Y': loc_Y}


# Normalize - inv
def normalize_inv(glob_train, glob_test, glob_Y):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)
	loc_train = loc_train.apply(lambda x: np.power(x, -1)) 
	loc_test = loc_test.apply(lambda x: np.power(x, -1)) 
	loc_Y = loc_Y.apply(lambda x: np.power(x, -1)) 

	return {'train': loc_train ,'test': loc_test, 'Y': loc_Y}


def imputationCity_knn(glob_train, glob_test):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)

	city_train_arr = loc_train["City"].unique()
	loc_test["City"] = loc_test["City"].map(lambda x: 'Other' if x not in city_train_arr else x)

	trainX_from_train = loc_train.drop(['City'], axis=1)
	trainY_from_train = loc_train['City']
	trainX_from_test = loc_test[loc_test.City != 'Other'].drop(['City'], axis=1)
	trainY_from_test = loc_test[loc_test.City != 'Other']['City']
	testY_from_test = loc_test[loc_test.City == 'Other'].drop(['City'], axis=1)
	
	my_train_X = pd.concat([trainX_from_train, trainX_from_test])
	my_train_Y = pd.concat([trainY_from_train, trainY_from_test])
	my_test_X = testY_from_test

	leA = preprocessing.LabelEncoder()
	leA.fit(pd.concat([my_train_X, my_test_X])['Type'])
	my_train_X['Type'] = leA.transform(my_train_X['Type'])
	my_test_X['Type'] = leA.transform(my_test_X['Type'])

	leB = preprocessing.LabelEncoder()
	leB.fit(pd.concat([my_train_X, my_test_X])['City Group'])
	my_train_X['City Group'] = leB.transform(my_train_X['City Group'])
	my_test_X['City Group'] = leB.transform(my_test_X['City Group'])


	neigh = KNeighborsClassifier(n_neighbors=2)
	neigh.fit(my_train_X, my_train_Y) 
	preds = neigh.predict(my_test_X)
	my_test_y = pd.DataFrame({'City': preds})
	my_test_y["City"].value_counts().plot(kind='bar')

	my_train_X['Type'] = leA.inverse_transform(my_train_X['Type'])
	my_test_X['Type'] = leA.inverse_transform(my_test_X['Type'])

	my_train_X['City Group'] = leB.inverse_transform(my_train_X['City Group'])
	my_test_X['City Group'] = leB.inverse_transform(my_test_X['City Group'])

	count = 0
	for index, row in loc_test.iterrows():
	    if row['City'] == 'Other':
	        loc_test.loc[index, "City"] = my_test_y["City"][count]
	        count += 1

	return {'train': loc_train ,'test': loc_test}


def imputationType_knn(glob_train, glob_test):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)

	type_train_arr = loc_train["Type"].unique()
	loc_test["Type"] = loc_test["Type"].map(lambda x: 'Other' if x not in type_train_arr else x)

	trainX_from_train = loc_train.drop(['Type'], axis=1)
	trainY_from_train = loc_train['Type']
	trainX_from_test = loc_test[loc_test.Type != 'Other'].drop(['Type'], axis=1)
	trainY_from_test = loc_test[loc_test.Type != 'Other']['Type']
	testY_from_test = loc_test[loc_test.Type == 'Other'].drop(['Type'], axis=1)
	
	my_train_X = pd.concat([trainX_from_train, trainX_from_test])
	my_train_Y = pd.concat([trainY_from_train, trainY_from_test])
	my_test_X = testY_from_test

	leA = preprocessing.LabelEncoder()
	leA.fit(pd.concat([my_train_X, my_test_X])['City'])
	my_train_X['City'] = leA.transform(my_train_X['City'])
	my_test_X['City'] = leA.transform(my_test_X['City'])

	leB = preprocessing.LabelEncoder()
	leB.fit(pd.concat([my_train_X, my_test_X])['City Group'])
	my_train_X['City Group'] = leB.transform(my_train_X['City Group'])
	my_test_X['City Group'] = leB.transform(my_test_X['City Group'])

	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(my_train_X, my_train_Y) 
	preds = neigh.predict(my_test_X)
	my_test_y = pd.DataFrame({'Type': preds})
	my_test_y["Type"].value_counts().plot(kind='bar')

	my_train_X['City'] = leA.inverse_transform(my_train_X['City'])
	my_test_X['City'] = leA.inverse_transform(my_test_X['City'])

	my_train_X['City Group'] = leB.inverse_transform(my_train_X['City Group'])
	my_test_X['City Group'] = leB.inverse_transform(my_test_X['City Group'])

	count = 0
	for index, row in loc_test.iterrows():
	    if row['Type'] == 'Other':
	        loc_test.loc[index, "Type"] = my_test_y["Type"][count]
	        count += 1

	return {'train': loc_train ,'test': loc_test}
	

# Only continuos variables from original data set
def data_format_1(glob_train, glob_test):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)
	cont_features = train.columns[train.dtypes != 'object']
	loc_train = loc_train[cont_features]
	loc_test = loc_test[cont_features]
	
	return {'train': loc_train ,'test': loc_test}


# Only continuos variables from original data set
# Droping variables as they as 75% times zeros P14 P15 P16 P17 P18 P24 P25 P26 P27 P30 P31 P32 P33 P34 P35 P36 P37
def data_format_2(glob_train, glob_test):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)
	cont_features = train.columns[train.dtypes != 'object']
	loc_train = loc_train[cont_features].drop(['P14','P15','P16','P17','P18','P24','P25','P26','P27','P30','P31','P32','P33','P34','P35','P36','P37'], axis=1)
	loc_test = loc_test[cont_features].drop(['P14','P15','P16','P17','P18','P24','P25','P26','P27','P30','P31','P32','P33','P34','P35','P36','P37'], axis=1)
	
	return {'train': loc_train ,'test': loc_test}


# All variables from original data set converted to numerical ; Date transformed into (days_open) and (month_opened)
def data_format_3(glob_train, glob_test):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)

	start_date = datetime.strptime('01/01/2015', '%m/%d/%Y')
	loc_train['days_open'] = loc_train['Open Date'].map(lambda x: (start_date - datetime.strptime(x, '%m/%d/%Y')).days + 1)
	loc_train['month_opened'] = loc_train['Open Date'].map(lambda x:datetime.strptime(x, '%m/%d/%Y').month)
	loc_train = loc_train.drop(['Open Date'], axis=1)

	loc_test['days_open'] = loc_test['Open Date'].map(lambda x: (start_date - datetime.strptime(x, '%m/%d/%Y')).days + 1)
	loc_test['month_opened'] = loc_test['Open Date'].map(lambda x:datetime.strptime(x, '%m/%d/%Y').month)
	loc_test = loc_test.drop(['Open Date'], axis=1)

	imputationType_res = imputationType_knn(loc_train, loc_test)
	loc_train = imputationType_res['train']
	loc_test = imputationType_res['test']
	
	imputationCity_res = imputationCity_knn(loc_train, loc_test)
	loc_train = imputationCity_res['train']
	loc_test = imputationCity_res['test']

	conversion_res = categorical2numerical(loc_train, loc_test)
	loc_train = conversion_res['train']
	loc_test = conversion_res['test']

	return {'train': loc_train ,'test': loc_test}


# All variables from original data set converted to numerical ; Date transformed into (days_open)
def data_format_4(glob_train, glob_test):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)

	data_format_2_res = data_format_2(glob_train, glob_test)
	loc_train = data_format_2_res['train'].drop(['month_opened'], axis=1)
	loc_test = data_format_2_res['test'].drop(['month_opened'], axis=1)

	return {'train': loc_train ,'test': loc_test}


# All variables from original data set converted to numerical ; Date transformed into (days_open)
# Droping variables as they as 75% times zeros P14 P15 P16 P17 P18 P24 P25 P26 P27 P30 P31 P32 P33 P34 P35 P36 P37
def data_format_5(glob_train, glob_test):
	loc_train = glob_train.copy(deep = True)
	loc_test = glob_test.copy(deep = True)

	data_format_3_res = data_format_3(glob_train, glob_test)
	loc_train = data_format_3_res['train'].drop(['P14','P15','P16','P17','P18','P24','P25','P26','P27','P30','P31','P32','P33','P34','P35','P36','P37'], axis=1)
	loc_test = data_format_3_res['test'].drop(['P14','P15','P16','P17','P18','P24','P25','P26','P27','P30','P31','P32','P33','P34','P35','P36','P37'], axis=1)
	
	return {'train': loc_train ,'test': loc_test}


def main():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')
	Y = train.revenue # training prediction values
	train = train.drop(['Id', 'revenue'], axis=1)
	test = test.drop(['Id'], axis=1 )


if __name__ == "__main__": main()