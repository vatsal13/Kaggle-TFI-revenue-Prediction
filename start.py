import pre_proc as pp_
import support_vector_regression as svr_
import random_forest_regression as rf_
import lasso_regression as ls_
import gradient_boosting_regression as gb_
import pandas as pd
import numpy as np

def main():
	#regression object = ls_ or svr_ or rf_ or gb_
	rg_ob = ls_
	# suffix for file name 
	f = 'ls'
	
	glob_train = pd.read_csv('train.csv')
	glob_test = pd.read_csv('test.csv')
	glob_Y = glob_train.revenue # training prediction values
	test_id = glob_test.Id

	glob_train = glob_train.drop(['Id', 'revenue'], axis=1)
	glob_test = glob_test.drop(['Id'], axis=1 )
	
	data0 = pp_.data_format_1(glob_train, glob_test)
	data1 = pp_.normalize_log(data0['train'], data0['test'], glob_Y)
	data2 = pp_.normalize_sqrt(data0['train'], data0['test'], glob_Y)

	rg_ob.run(data0['train'], data0['test'], glob_Y, test_id, f+'1.csv', transform=None, standardize=False, n_pca=None)
	rg_ob.run(data0['train'], data0['test'], glob_Y.apply(lambda x: np.log(x+1)), test_id, f+'2.csv', transform='log', standardize=False, n_pca=None)
	rg_ob.run(data1['train'], data1['test'], data1['Y'], test_id, f+'3.csv', transform='log', standardize=False, n_pca=None)
	rg_ob.run(data2['train'], data2['test'], data2['Y'], test_id, f+'4.csv', transform='sqrt',standardize=False, n_pca=None)


	data3 = pp_.data_format_2(glob_train, glob_test)
	data4 = pp_.normalize_log(data3['train'], data3['test'], glob_Y)
	data5 = pp_.normalize_sqrt(data3['train'], data3['test'], glob_Y)

	rg_ob.run(data3['train'], data3['test'], glob_Y, test_id, f+'5.csv', transform=None, standardize=True, n_pca=None)
	rg_ob.run(data3['train'], data3['test'], glob_Y.apply(np.sqrt), test_id, f+'6.csv', transform='sqrt', standardize=False, n_pca=None)
	rg_ob.run(data3['train'], data3['test'], glob_Y.apply(lambda x: np.log(x+1)), test_id, f+'7.csv', transform='log', standardize=False, n_pca=None)
	rg_ob.run(data3['train'], data3['test'], glob_Y.apply(lambda x: np.log(x+1)), test_id, f+'8.csv', transform='log', standardize=False, n_pca=12)
	rg_ob.run(data3['train'], data3['test'], glob_Y.apply(lambda x: np.log(x+1)), test_id, f+'9.csv', transform='log', standardize=False, n_pca=16)
	rg_ob.run(data4['train'], data4['test'], data4['Y'], test_id, f+'10.csv', transform='log', standardize=False, n_pca=None)
	rg_ob.run(data4['train'], data4['test'], data4['Y'], test_id, f+'11.csv', transform='log', standardize=False, n_pca=12)
	rg_ob.run(data4['train'], data4['test'], data4['Y'], test_id, f+'12.csv', transform='log', standardize=False, n_pca=16)
	rg_ob.run(data5['train'], data5['test'], data5['Y'], test_id, f+'13.csv', transform='sqrt', standardize=False, n_pca=None)


	data9 = pp_.data_format_4(glob_train, glob_test)
	data10 = pp_.normalize_log(data9['train'], data9['test'], glob_Y)
	data11 = pp_.normalize_sqrt(data9['train'], data9['test'], glob_Y)

	
	rg_ob.run(data9['train'], data9['test'], glob_Y, test_id, f+'14.csv', transform=None, standardize=True, n_pca=None)
	rg_ob.run(data9['train'], data9['test'], glob_Y, test_id, f+'15.csv', transform=None, standardize=True, n_pca=15)
	rg_ob.run(data9['train'], data9['test'], glob_Y.apply(np.sqrt), test_id, f+'16.csv', transform='sqrt', standardize=True, n_pca=15)
	rg_ob.run(data9['train'], data9['test'], glob_Y.apply(lambda x: np.log(x+1)), test_id, f+'17.csv', transform='log', standardize=True, n_pca=None)
	rg_ob.run(data9['train'], data9['test'], glob_Y.apply(lambda x: np.log(x+1)), test_id, f+'18.csv', transform='log', standardize=True, n_pca=15)
	rg_ob.run(data10['train'], data10['test'], data10['Y'], test_id, f+'19.csv', transform='log', standardize=False, n_pca=None)
	rg_ob.run(data10['train'], data10['test'], data10['Y'], test_id, f+'20.csv', transform='log', standardize=False, n_pca=15)

	data12 = pp_.data_format_5(glob_train, glob_test)
	data13 = pp_.normalize_log(data12['train'], data12['test'], glob_Y)
	data14 = pp_.normalize_sqrt(data12['train'], data12['test'], glob_Y)

	rg_ob.run(data12['train'], data12['test'], glob_Y, test_id, f+'21.csv', transform=None, standardize=True, n_pca=12)
	rg_ob.run(data12['train'], data12['test'], glob_Y, test_id, f+'22.csv', transform=None, standardize=True, n_pca=16)

	rg_ob.run(data12['train'], data12['test'], glob_Y.apply(np.sqrt), test_id, f+'23.csv', transform='sqrt', standardize=True, n_pca=12)
	rg_ob.run(data12['train'], data12['test'], glob_Y.apply(np.sqrt), test_id, f+'24.csv', transform='sqrt', standardize=True, n_pca=16)

	rg_ob.run(data12['train'], data12['test'], glob_Y.apply(lambda x: np.log(x+1)), test_id, f+'25.csv', transform='log', standardize=True, n_pca=None)
	rg_ob.run(data12['train'], data12['test'], glob_Y.apply(lambda x: np.log(x+1)), test_id, f+'26.csv', transform='log', standardize=True, n_pca=12)
	rg_ob.run(data12['train'], data12['test'], glob_Y.apply(lambda x: np.log(x+1)), test_id, f+'27.csv', transform='log', standardize=True, n_pca=16)
	
	rg_ob.run(data13['train'], data13['test'], data13['Y'], test_id, f+'28.csv', transform='log', standardize=False, n_pca=None)
	rg_ob.run(data13['train'], data13['test'], data13['Y'], test_id, f+'29.csv', transform='log', standardize=False, n_pca=12)
	rg_ob.run(data13['train'], data13['test'], data13['Y'], test_id, f+'30.csv', transform='log', standardize=False, n_pca=16)

	return ''


if __name__ == "__main__": main()