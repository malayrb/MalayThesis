##############################################################################
# To use the script: python filename.py file1.csv
# Make sure that the sklearn module is 0.17 or higher. If not, you can install it from https://pypi.python.org/pypi/scikit-learn/0.17.1#downloads. To install runpython setup.py build
#sudo python setup.py install 
##############################################################################
###################### Below modules should be installed ###################
import csv
import numpy as np
from sys import argv
import pickle
from sklearn import *
from sklearn.neural_network import MLPRegressor
from pylab import *
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
##############################Reading the csv input######################
tester=np.loadtxt(open(argv[1],"rb"),delimiter=",",skiprows=1)
t=list(tester)
test=np.array(t).astype('float')

with open(argv[1]) as h:
    ques_t = csv.reader(h, delimiter=',', skipinitialspace=True)
    one_row_t = next(ques_t)
y_test = test[:,0]
X_test = np.delete(test, 0, axis=1)

######################Data preprocessing###########################
pickle_in = open("scaler.pickle","rb")
scaler = pickle.load(pickle_in)
pickle_in.close()

X_test = scaler.transform(X_test)
######################Neural_fit######################################
pickle_in = open("model1.pickle","rb")
mlp = pickle.load(pickle_in)
pickle_in.close()

predict_test = mlp.predict(X_test)

X_test_2 = scaler.inverse_transform(X_test)

# If the data contains the experimental value, un-comment the below lines to check R^2

#r2_test = r2_score(y_test,predict_test)  

with open('log_file.txt','w') as f:
#  f.write("#r2_test = " +str(r2_test)+'\n\n')
#  f.write('\n')
  np.savetxt(f, np.c_[predict_test,y_test,X_test_2], fmt='%f', header='#calc_test Exp_test ', comments='' )


## To plot the result with respect to the experimental value, un-comment the below lines 
##########################Plotting the result#################
#fig, ax = plt.subplots()
#ax.scatter(y_test,predict_test,s=80,color='#14ad30',marker='D', alpha=0.9, label=r'$R^2_{test}=$'+str("%.3f" % r2_test) )
#ax.plot([0,250], [0,250], 'k--', lw=2)
#
#ax.legend(loc='lower right', frameon=False, fontsize=20, markerscale=0)
#plt.xticks(fontsize=17)
#plt.yticks(fontsize=17)
#
#ax.set_xlabel(r'Experimental MIC ($\mu$g/ml)', fontsize=20)
#ax.set_ylabel(r'Calculated MIC ($\mu$g/ml)', fontsize=20)
#savefig('plot.pdf',bbox_inches='tight')
