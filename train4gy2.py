import numpy as np
import scipy.io 
import theano.tensor as T
import lasagne
import string
import theano
import BatchNormLayer
batch_norm = BatchNormLayer.batch_norm

############# TrainData and TestData Preparation #############

# trainx: 780*50*6 
# trainy: 780*52 or 780
# mask_train: 780*50
loaddata = scipy.io.loadmat('../dataset/Data2.mat')
traindata = loaddata['TrainData']
testdata = loaddata['TestData']

trainx = []
trainy = []

for i in xrange(traindata.shape[0]):
	for j in xrange(traindata.shape[1]):
		trainx.append(traindata[i][j])
		trainy.append(i)

trainx = np.asarray(trainx)
trainy = np.asarray(trainy)
mask_train = np.ones([trainx.shape[0], trainx.shape[1]])
#print trainy.shape
#print trainy

# testx: 624*20*6
# testy: 624*52 or 624
# mask_test: 624*20
testx = []
testy = []

for i in xrange(testdata.shape[0]):
	for j in xrange(testdata.shape[1]):
		testx.append(testdata[i][j])
		testy.append(i)

testx = np.asarray(testx)
testy = np.asarray(testy)
mask_test = np.ones([testx.shape[0], testx.shape[1]])
#print testy.shape
#print testy

##########  Initialize Parameters ########
N_BATCH = 52
MAX_LENGTH = 20
features_num = 6
GRAD_CLIP = 100
TOL = 1e-5
LEARNING_RATE = 1e-2
N_HIDDEN = 120
num_epochs = 500

##########  Build Model for Prediction #######################
mask = np.ones([N_BATCH, MAX_LENGTH])
target_values = T.ivector('target_output')
sym_mask = T.matrix('mask')

print("Building network ...")

l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, features_num))
#l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
#### Bidirectional RNN layers ###
l_forward = lasagne.layers.GRULayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP, only_return_final=True)
        #W_in_to_hid=lasagne.init.HeUniform(),
        #W_hid_to_hid=lasagne.init.HeUniform(),
        #nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)

l_backward = lasagne.layers.GRULayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP, only_return_final=True, backwards=True)
        #W_in_to_hid=lasagne.init.HeUniform(),
        #W_hid_to_hid=lasagne.init.HeUniform(),
        #nonlinearity=lasagne.nonlinearities.tanh,
        #only_return_final=True, backwards=True)


l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
l_concat = batch_norm(l_concat)

###  Softmax for prediction output ######
l_fc1 = lasagne.layers.DenseLayer(
	    lasagne.layers.dropout(l_concat, p=.5),
        num_units=100, nonlinearity=lasagne.nonlinearities.rectify)
l_fc1 = batch_norm(l_fc1)

l_out = lasagne.layers.DenseLayer(
	    lasagne.layers.dropout(l_fc1, p=.5),
        num_units=52, nonlinearity=lasagne.nonlinearities.softmax)

#network_output = network_output.reshape((-1, 52))

########### Debug ###########
all_layers = lasagne.layers.get_all_layers(l_out)
num_params = lasagne.layers.count_params(l_out)
print("  number of parameters: %d" % num_params)
print("  layer output shapes:")
for layer in all_layers:
	name = string.ljust(layer.__class__.__name__, 32)
	print("    %s %s" % (name, lasagne.layers.get_output_shape(layer)))

############## Compile the model, calculate cost, update ######################
#########  #######
network_output = lasagne.layers.get_output(l_out)

out_train = lasagne.layers.get_output(l_out, mask = sym_mask, deterministic=False)

out_test = lasagne.layers.get_output(l_out, mask = sym_mask, deterministic=True)

cost = T.nnet.categorical_crossentropy(T.clip(network_output, TOL, 1-TOL), target_values)
cost = cost.mean()
# Retrieve all parameters from the network
all_params = lasagne.layers.get_all_params(l_out, trainable=True)

all_grads = T.grad(cost, all_params)

updates, norm_calc = lasagne.updates.total_norm_constraint(all_grads, max_norm= 20, return_norm=True)

# Compute SGD updates for training
print("Computing updates ...")
#updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
updates = lasagne.updates.adam(cost, all_params, LEARNING_RATE)

print("Compiling functions ...")
train = theano.function([l_in.input_var, target_values, sym_mask], \
	                    [cost, out_train], updates=updates, \
	                    on_unused_input='ignore', allow_input_downcast=True)
#compute_cost = theano.function([l_in.input_var, target_values, l_mask.input_var], cost)
test = theano.function([l_in.input_var, target_values, sym_mask], \
                        [cost, out_test], on_unused_input='ignore', allow_input_downcast=True)

print("Training ...")
seq_names = np.arange(0, trainx.shape[0])
np.random.shuffle(seq_names)
trainx = trainx[seq_names]
trainy = trainy[seq_names]  
mask_train = mask_train[seq_names]
cost_train = []
out_train = []
EPOCH_SIZE_TRAIN = trainx.shape[0]// N_BATCH

seq_names_test = np.arange(0, testx.shape[0])
np.random.shuffle(seq_names_test)
testx = testx[seq_names_test]
testy = testy[seq_names_test]  
mask_test = mask_test[seq_names_test]
cost_test = []
out_test = []
EPOCH_SIZE_TEST = testx.shape[0]// N_BATCH

for epoch in range(num_epochs):
	### Train part ###
    out_train_epoch = []
    cost_train_epoch = 0
    for i in xrange(EPOCH_SIZE_TRAIN):
    	idx = xrange(i*N_BATCH, (i+1)*N_BATCH)
    	x_batch = trainx[idx]
    	y_batch = trainy[idx]
    	mask_batch = mask_train[idx]

    	cost_train_batch, out_train_batch = train(x_batch, y_batch, mask_batch)
    	out_train_epoch.append(out_train_batch)
    	cost_train_epoch += cost_train_batch

    out_train_epoch = np.asarray(out_train_epoch)
    out_train_epoch = out_train_epoch.reshape(-1, 52)    
    out_train.append(out_train_epoch)
    cost_train.append(cost_train_epoch)
    ### Accuracy calculation ####
    sum = 0
    for i in xrange(out_train_epoch.shape[0]):
    	predict_label = np.argmax(out_train_epoch[i,:])
    	if predict_label == trainy[i]:
    		sum += 1
    accuracy = float(sum)/float(out_train_epoch.shape[0])    
    print("Epoch {} train cost = {}".format(epoch, cost_train_epoch))
    print("Epoch {} train accuracy = {}".format(epoch, accuracy))

    ### Inference Part ###
    out_test_epoch = []
    cost_test_epoch = 0
    for i in range(EPOCH_SIZE_TEST):
		idx = range(i*N_BATCH, (i+1)*N_BATCH)
		x_batch = testx[idx]
		y_batch = testy[idx]
		mask_batch = mask_test[idx]

		cost_test_batch, out_test_batch = test(x_batch, y_batch, mask_batch)
		out_test_epoch.append(out_test_batch)
		cost_test_epoch += cost_test_batch

    out_test_epoch = np.asarray(out_test_epoch)
    out_test_epoch = out_test_epoch.reshape(-1, 52)
    out_test.append(out_test_epoch)
    cost_test.append(cost_test_epoch)
    #print out_test_epoch.shape
    #print out_test_epoch
    ### Accuracy calculation ####
    sum = 0
    for i in xrange(out_test_epoch.shape[0]):
    	predict_label = np.argmax(out_test_epoch[i,:])
    	if predict_label == testy[i]:
    		sum += 1
    accuracy = float(sum)/float(out_test_epoch.shape[0])
    
    print("Epoch {} test cost = {}".format(epoch, cost_test_epoch))
    print("Epoch {} test accuracy = {}".format(epoch, accuracy))
