import pickle

#Load pickle file
input_file = 'a3c_weights.pkl' #sys.argv[1]
pkl_file = open(input_file, 'rb')
pkl = pickle.load(pkl_file)
weights = pkl["weights"]
pkl_file.close()
