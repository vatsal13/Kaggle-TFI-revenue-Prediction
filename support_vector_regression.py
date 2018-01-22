from sklearn.preprocessing import Normalizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import pandas as pd
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import numpy as np

def submit(model, test, test_id, file_name, transform):
	preds = model.predict(test)
	submission = pd.DataFrame({'Id':test_id, 'Prediction': preds})
	if transform == 'inv':
		submission["Prediction"] = submission["Prediction"].map(lambda x: np.power(x, -1))
	if transform == 'sqrt':
		submission["Prediction"] = submission["Prediction"].map(lambda x: np.power(x, 2))
	if transform == 'log':
		submission["Prediction"] = submission["Prediction"].map(lambda x: np.exp(x) - 1)
	submission.to_csv(file_name, index=False)


def run(train, test, Y, test_id, file_name, transform=None, standardize=False, n_pca=None):

	pipe = Pipeline([('clf', SVR())])
	
	if standardize:
		pipe = Pipeline([('scl', StandardScaler()),('clf', SVR())])

	if n_pca:
		pca = PCA(n_components=n_pca)
		train = pca.fit_transform(train)
		test = pca.transform(test)

	param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
	gamma_range = [0.001, 0.1, 1, 10]
	param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear','rbf','sigmoid']},
				  {'clf__C': param_range, 'clf__gamma': gamma_range, 'clf__kernel': ['rbf', 'sigmoid']}]

	gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10)
	
	gs = gs.fit(train, Y)
	
	submit(gs, test, test_id, file_name, transform)
	
	print ">>> ", file_name
	print(gs.best_score_)
	print "---> ", np.sqrt(-gs.best_score_)
	print(gs.best_params_)
	print('\n')