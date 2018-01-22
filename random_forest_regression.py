from sklearn.preprocessing import Normalizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

	pipe = Pipeline([('rf', RandomForestRegressor())])
	
	if standardize:
		pipe = Pipeline([('scl', StandardScaler()),('rf', RandomForestRegressor())])
	
	if n_pca:
		pca = PCA(n_components=n_pca)
		train = pca.fit_transform(train)
		test = pca.transform(test)

	estimators_range = [50, 500, 1000, 1500]
	split_range = [2, 3]
	param_grid = [{'rf__n_estimators': estimators_range, 'rf__min_samples_split': split_range} ]

	gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10)
	
	gs = gs.fit(train, Y)
	
	submit(gs, test, test_id, file_name, transform)
	print ">>> ", file_name
	print(gs.best_score_)
	print "---> ", np.sqrt(-gs.best_score_)
	print(gs.best_params_)
	print('\n')