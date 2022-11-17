import pickle
from sklearn.preprocessing import RobustScaler
import numpy as np
# Open the file in binary mode
with open('scale.pkl', 'rb') as file:
	
	# Call load method to deserialze
	myvar = pickle.load(file)
print(myvar)
TR = np.array([56,1,1,1,2,1,1,1,5,1,261,1,1,0.6987,-36.4])
TR = TR.reshape(1,-1)
print(TR)
test_record = myvar.transform(TR)
print(test_record)
    # test_record1=standardscalar.fit_transform(TR)
# print(test_record1)
