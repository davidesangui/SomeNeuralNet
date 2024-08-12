import random
import time
try:
    import cupy as np
    import numpy as cpunp
except Exception:
    import numpy as np
    cpunp=np



# Implementation of few most common things for making deep learning in a supervised fashion.
# There is a lot of space for improvements, and probably a lot of naive coding, 
# but my main purpose was implementing for learning neural networks, not how to make and a super efficient and compact library.


################################################ COST FUNCTIONS

def squared_delta_cost(last_layer_activations,expected_activations):
    return np.sum((last_layer_activations-expected_activations)**2)

def cross_entropy_loss(last_layer_activations,expected_activations):
    return -1*np.sum(expected_activations*np.log(last_layer_activations))

###########################################################################

################################################ REGULARIZATION FUNCTIONS

def l2_regularization(net):
    return net.reg_hyper*sum([np.sum(layer.wmatrix**2) for layer in net.architecture])

###########################################################################

################################################ EXPECTED ACTIVATIONS

def sigmoid_expected_activations(label,length):
    return np.array([1 if i==label else 0 for i in range(length)])

###########################################################################

################################################ ACTIVATION FUNCTIONS

def sig_activation(zetas): 
    return 1/(1+np.exp(-1*(zetas)))

def relu_activation(zetas): 
    return(np.maximum(0,zetas))


def linear_activation(zetas):  
    return(zetas)

def tanh_activation(zetas): 
    return np.tanh(zetas)

###########################################################################

################################################ DERIVATIVE FUNCTIONS

def squared_delta_cost_derivative(last_layer_activations,expected_activations):
    return 2*(last_layer_activations-expected_activations)

def cross_entropy_loss_derivative(last_layer_activations,expected_activations):
    return(last_layer_activations-expected_activations)

def sigmoid_derivative(ndarr):
    return ndarr*(1-ndarr)

def relu_derivative(ndarr):
    return(np.where(ndarr<=0,0,1))

def linear_derivative(ndarr):
    return np.ones(ndarr.shape)


############################################################################

################################################ MISCELLANEOUS

def load_mnist(mnist_file,start=0,howmany=-1):  # very slow. better save as numpy objects
    """load file in mnist format. start=0 and howmany=-1 to load all file."""
    f=open(mnist_file)
    inputs=[]
    labels=[]
    c=0
    for line in f:
        if start>0:
            start-=1
            continue
        c+=1
        a=line.strip().split(',')
        labels.append(int(a[0]))
        inputs.append(cpunp.array([float(x) for x in a[1:]]))
        if c==howmany:
            break
    return(cpunp.array(inputs),cpunp.array(labels))

def randomizer(data):
    """randomize training set maintaining input-label pairing"""
    toshuffle=[]
    for i in range(len(data[0])):
        toshuffle.append((data[0][i],data[1][i]))
    random.shuffle(toshuffle)
    inp,lab=[],[]
    for x,y in toshuffle:
        inp.append(x)
        lab.append(y)
    return(cpunp.array(inp),cpunp.array(lab))

def glorot_uniform(fanin,fanout):
    return np.sqrt(6)/np.sqrt(fanin+fanout)

def he_uniform(fanin):
    return(np.sqrt(6 / fanin))

def accuracy_ontest(truelabs,predicted_labs):
    """count number of equal labels between truth and prediction"""
    giusti=0
    for i in range(len(truelabs)):
        if predicted_labs[i]==truelabs[i]:
            giusti+=1
    return giusti/len(truelabs)

def tokenizer(file,torem, punct): 
    """Transforms the whole text in the file in a sequence of numbers.
       Any punctuation is considered as a single word.
        - torem is a list of characters to remove when parsing the text
        - punct is a list of punctuation characters"""
    lines=[x.strip() for x in open(file,encoding='utf-8').readlines()]
    newlines=[]
    for line in lines:
        for p in punct:
            line=line.replace(p,' '+p+' ')
        newlines.append(line)
    words=[]
    for line in newlines:
        for w in line.split(' '):
            if w:
                for r in torem:
                    w=w.strip(r)
                words.append(w.lower())
        words.append('\n')
    ind=0
    word_index={}
    index_word={}
    inps=[]
    for i in words:
        if i in word_index:
            inps.append(word_index[i])
        else:
            word_index[i]=ind
            index_word[ind]=i
            inps.append(ind)
            ind+=1
    return index_word,word_index,cpunp.array(inps)


def lrupdown(nn,jump,upperlim,lowerlim,upordown): 
    """helper function for learning rate warmup"""
    if upordown=='up':
        return(min(upperlim,nn.lrate/jump))
    else:
        return(max(lowerlim,nn.lrate*jump))

############################################################################

# There are two type of classes: the layers and the network. 
#
# All the layers have the same key methods. In retrospect, an abstract class would have been useful here. 
#
# - activate(args): take input and return layer output
# - test_activate(args): same but at test time
# - adam_move_toward_gradients(args): perform the learning step using the adam algorithm 
# - update_gradients(args): calculate and update the layer gradients before doing the learning step
# - save_weights(args): Very badly-written way to save layer weights to file. This was usually the last thing done hence I never had the will to make it better.
# 
# The network is essentially a sequence of layers. Its methods are:
# - wave_from_inputs(args): take input and return list of all layers activations
# - cost_and_backpropagation(args): take input and labels, get network output, calculate cost, perform backpropagation storing gradients without update.
# - update_parameters(): update weights using stored gradients
# - linear_batch_gd(args): minibatch gradient descent 
# - predict(args): take input, return class label.
# - load_weights(args): Even worse than save_weights (as it can be seen from how much is long), but I am a bit lazy.
#
# Sorry for the absence of comments that may help to read the codes. 
# All the present commments were meant for my reading.
 
 
class fully_connected_layer:
    def __init__(self,n_weights,n_neurons,intorno,act_func,act_derivative):
        self.W=np.random.uniform(-intorno,intorno,(n_neurons,n_weights))
        self.B=np.zeros((n_neurons,))
        self.Velw=np.zeros((n_neurons,n_weights))
        self.RMSPw=np.zeros((n_neurons,n_weights))
        self.Velb=np.zeros((n_neurons,))
        self.RMSPb=np.zeros((n_neurons,))
        self.act_func=act_func
        self.act_derivative=act_derivative
        self.Gradw=np.zeros((n_neurons,n_weights))
        self.Gradb=np.zeros((n_neurons,))
    
    def activate(self,inps): 
        zetas=np.dot(inps,self.W.T)+self.B
        return self.act_func(zetas)

    def test_activate(self,inps):
        return self.activate(inps)

    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        self.Velw=mrate*self.Velw+(1-mrate)*self.Gradw # update w vel
        self.Velb=mrate*self.Velb+(1-mrate)*self.Gradb # update b vel
        self.RMSPw=rmsprate*self.RMSPw+(1-rmsprate)*self.Gradw**2 # update w rmsp
        self.RMSPb=rmsprate*self.RMSPb+(1-rmsprate)*self.Gradb**2 # update b rmsp
        self.W+=-1*lrate*((self.Velw/(1-mrate))/(np.sqrt(self.RMSPw/(1-rmsprate))+small_constant)) # update w
        self.B+=-1*lrate*((self.Velb/(1-mrate))/(np.sqrt(self.RMSPb/(1-rmsprate))+small_constant)) # update b
        self.Gradw*=0 # set gradients to zero
        self.Gradb*=0

    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor): # set average_factor to one in sequences learning
        rawgrads=self.act_derivative(acts_of_lay)*act_grad
        self.Gradw+=np.dot(rawgrads.T,inps_to_lay)/average_factor
        self.Gradb+=np.sum(rawgrads,0)/average_factor # depending on cupy version, here an error may be raised. The following if/else statement should be used incase.
        #if len(self.Gradb)!=1:  
        #    self.Gradb+=np.sum(rawgrads,0)/average_factor          
        #else:
        #    self.Gradb+=np.sum(rawgrads)/average_factor
        return np.dot(rawgrads,self.W) # return next activation gradients  
         
    def save_weights(self,file_object):
        print('fully_connected_layer',file=file_object)
        if self.act_func is relu_activation:
            print('relu_activation',file=file_object)
        elif self.act_func is sig_activation:
            print('sigmoid_activation',file=file_object)
        elif self.act_func is linear_activation:
            print('linear_activation',file=file_object)
        elif self.act_func is tanh_activation:
            print('tanh_activation',file=file_object)
        print(';'.join([','.join([str(x) for x in neuwei]) for neuwei in self.W]),file=file_object)
        print(';'.join([str(x) for x in self.B]),file=file_object)

class batchnorm_layer:
    def __init__(self,n_features,mrate=0.9,small_constant=1e-3): # n_features e' il numero di neuroni del layer precedente
        self.mrate=mrate
        self.moving_mean=np.zeros((1,n_features))
        self.moving_var=np.ones((1,n_features))
        self.gamma=np.ones((1,n_features))
        self.beta=np.zeros((1,n_features))
        self.Velgamma=np.zeros((1,n_features))
        self.Velbeta=np.zeros((1,n_features))
        self.RMSPgamma=np.zeros((1,n_features))
        self.RMSPbeta=np.zeros((1,n_features))
        self.Gradgamma=np.zeros((1,n_features))
        self.Gradbeta=np.zeros((1,n_features))
        self.small_constant=small_constant
        self.standardized_inps=None # this will be needed to store the stand_inps for backpropagation
        self.varsofnow=None # this will be needed to store the stand_inps for backpropagation

    def activate(self,inps):
        means,vars=np.mean(inps,0),np.var(inps,0) # mean and std along the columns. the result is array as long as number of input dimensions
        self.varsofnow=vars
        self.moving_mean=self.mrate*self.moving_mean+(1-self.mrate)*means
        self.moving_var=self.mrate*self.moving_var+(1-self.mrate)*vars
        standardized_inps=(inps-means)/np.sqrt(vars+self.small_constant) # small_constant for numerical stability
        self.standardized_inps=standardized_inps
        return self.gamma*standardized_inps+self.beta

    def test_activate(self,inps):
        standardized_inps=(inps-self.moving_mean)/np.sqrt(self.moving_var+self.small_constant)
        return self.gamma*standardized_inps+self.beta

    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        self.Velgamma=mrate*self.Velgamma+(1-mrate)*self.Gradgamma # update w vel
        self.Velbeta=mrate*self.Velbeta+(1-mrate)*self.Gradbeta # update b vel
        self.RMSPgamma=rmsprate*self.RMSPgamma+(1-rmsprate)*self.Gradgamma**2 # update w rmsp
        self.RMSPbeta=rmsprate*self.RMSPbeta+(1-rmsprate)*self.Gradbeta**2 # update b rmsp
        self.gamma+=-1*lrate*((self.Velgamma/(1-mrate))/(np.sqrt(self.RMSPgamma/(1-rmsprate))+small_constant)) # update w
        self.beta+=-1*lrate*((self.Velbeta/(1-mrate))/(np.sqrt(self.RMSPbeta/(1-rmsprate))+small_constant)) # update b
        self.Gradgamma*=0 # set gradients to zero
        self.Gradbeta*=0

    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor):
        tmpgradgamma=np.sum(act_grad*self.standardized_inps,0)
        tmpgradbeta=np.sum(act_grad,0)
        self.Gradgamma+=tmpgradgamma 
        self.Gradbeta+=tmpgradbeta
        # return next activation gradients
        return self.gamma/average_factor*(1/np.sqrt(self.varsofnow+self.small_constant))*(-tmpgradgamma*self.standardized_inps+average_factor*act_grad-np.dot(np.ones((average_factor,1)),np.reshape(tmpgradbeta,(1,len(tmpgradbeta)))))

    def save_weights(self,file_object):
        print('batchnorm_layer',file=file_object)
        print(print(';'.join([str(x) for x in self.gamma[0]]),file=file_object))
        print(print(';'.join([str(x) for x in self.beta[0]]),file=file_object))
        print(print(';'.join([str(x) for x in self.moving_mean[0]]),file=file_object))
        print(print(';'.join([str(x) for x in self.moving_var[0]]),file=file_object))
        print(self.small_constant,file=file_object)

class layernorm_layer:
    def __init__(self,n_features,small_constant=1e-3): # n_features e' il numero di neuroni del layer precedente
        self.gamma=np.ones((1,n_features))
        self.beta=np.zeros((1,n_features))
        self.Velgamma=np.zeros((1,n_features))
        self.Velbeta=np.zeros((1,n_features))
        self.RMSPgamma=np.zeros((1,n_features))
        self.RMSPbeta=np.zeros((1,n_features))
        self.Gradgamma=np.zeros((1,n_features))
        self.Gradbeta=np.zeros((1,n_features))
        self.small_constant=small_constant
        self.standardized_inps=None # this will be needed to store the stand_inps for backpropagation
        self.stdsofnow=None # this will be needed to store the stand_inps for backpropagation
        self.meansofnow=None # this will be needed to store the stand_inps for backpropagation
    def activate(self,inps):
        means,stds=np.reshape(np.mean(inps,1),(len(inps),1)),np.reshape(np.var(inps,1),(len(inps),1)) # mean and std along the columns. the result is array as long as number of input dimensions
        self.stdsofnow=stds #### in realta' sono variances perche' viene piu' precisa la backprop, ma poca voglia di cambiare nome
        self.meansofnow=means
        standardized_inps=(inps-means)/np.sqrt(stds+self.small_constant) # small_constant for numerical stability
        self.standardized_inps=standardized_inps
        return self.gamma*standardized_inps+self.beta    
    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        self.Velgamma=mrate*self.Velgamma+(1-mrate)*self.Gradgamma # update w vel
        self.Velbeta=mrate*self.Velbeta+(1-mrate)*self.Gradbeta # update b vel
        self.RMSPgamma=rmsprate*self.RMSPgamma+(1-rmsprate)*self.Gradgamma**2 # update w rmsp
        self.RMSPbeta=rmsprate*self.RMSPbeta+(1-rmsprate)*self.Gradbeta**2 # update b rmsp
        self.gamma+=-1*lrate*((self.Velgamma/(1-mrate))/(np.sqrt(self.RMSPgamma/(1-rmsprate))+small_constant)) # update w
        self.beta+=-1*lrate*((self.Velbeta/(1-mrate))/(np.sqrt(self.RMSPbeta/(1-rmsprate))+small_constant)) # update b
        self.Gradgamma*=0 # set gradients to zero
        self.Gradbeta*=0
    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor):
        self.Gradgamma+=np.sum(act_grad*self.standardized_inps,0) # qui e sotto forse ci fa l'average factor, forse no, nel dubbio provare ogni volta e vedere cosa e' meglio
        self.Gradbeta+=np.sum(act_grad,0)
        dLnorm=act_grad*self.gamma
        lnorm=self.standardized_inps
        n_in=inps_to_lay.shape[-1]
        inps_actgrad = (n_in * dLnorm- dLnorm.sum(axis=1, keepdims=True)- lnorm * (dLnorm * lnorm).sum(axis=1, keepdims=True)) / (n_in * np.sqrt(self.stdsofnow+self.small_constant))
        return inps_actgrad
        # return next activation gradients
    def save_weights(self,file_object):
        print('layernorm_layer',file=file_object)
        print(print(';'.join([str(x) for x in self.gamma[0]]),file=file_object))
        print(print(';'.join([str(x) for x in self.beta[0]]),file=file_object))
        print(self.small_constant,file=file_object)

class dropout_layer:
    def __init__(self,n_features,prob):
        self.prob=prob
        self.nw=n_features
    def activate(self,inps):
        self.W=np.random.choice([0,1/(1-self.prob)],size=(1,self.nw),p=[self.prob,1-self.prob])
        return inps*self.W
    def test_activate(self,inps):
        return inps 
    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor):
        return self.W*act_grad
    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        pass
    def save_weights(self,file_object):
        print('dropout_layer',file=file_object)
        print(self.prob,file=file_object)
        print(self.nw,file=file_object)

class residual_block:
    def __init__(self,layers):
        self.components=[x for x in layers]
        self.allacts=[] # salva in attributo le attivazioni di ciascun layer per la backprop e anche l'input al block
    def activate(self,inps):
        inblock_inps=inps
        self.allacts.append(inblock_inps)
        for layer in self.components:
            inblock_inps=layer.activate(inblock_inps)
            self.allacts.append(inblock_inps)
        return inblock_inps+inps
    def test_activate(self,inps):
        inblock_inps=inps
        self.allacts.append(inblock_inps)
        for layer in self.components:
            inblock_inps=layer.test_activate(inblock_inps)
            self.allacts.append(inblock_inps)
        return inblock_inps+inps
    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        for layer in self.components:
            layer.adam_move_toward_gradients(lrate,mrate,rmsprate,small_constant=small_constant)
    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor): # set average_factor to one in sequence learning
        initial_act_grad=act_grad
        for i in range(len(self.components)-1,-1,-1):
            act_grad=self.components[i].update_gradients(self.allacts[i],self.allacts[i+1],act_grad,average_factor)
        self.allacts=[]
        return act_grad+initial_act_grad # return next activation gradients  
    def save_weights(self,file_object):
        print('residual_block',file=file_object)
        print(len(self.components),file=file_object)
        for layer in self.components:
            layer.save_weights(file_object)
    def load_weights(self,f):
        ncomp=int(f.readline().strip())
        line=f.readline().strip()
        while ncomp>0:
            if line=='fully_connected_layer':
                layer=fully_connected_layer(1,1,1,1,1) # inizializza a caso il layer
                act_fun=f.readline().strip()
                if act_fun=='sigmoid_activation':
                    layer.act_func=sig_activation
                    layer.act_derivative=sigmoid_derivative
                elif act_fun=='relu_activation':
                    layer.act_func=relu_activation
                    layer.act_derivative=relu_derivative
                elif act_fun=='linear_activation':
                    layer.act_func=linear_activation
                    layer.act_derivative=linear_derivative
                weights=f.readline().strip()
                layer.W=np.array([[float(x) for x in neu.split(',')] for neu in weights.split(';')])
                layer.B=np.array([float(x) for x in f.readline().strip().split(';')])
                n_neurons=len(layer.W)
                n_weights=len(layer.W[0])
                layer.Velw=np.zeros((n_neurons,n_weights))
                layer.RMSPw=np.zeros((n_neurons,n_weights))
                layer.Velb=np.zeros((n_neurons,))
                layer.RMSPb=np.zeros((n_neurons,))
                layer.Gradw=np.zeros((n_neurons,n_weights)) 
                layer.Gradb=np.zeros((n_neurons,))
                line=f.readline().strip()
            elif line=='batchnorm_layer':
                layer=batchnorm_layer(1)
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.gamma=np.reshape(tmp,(1,len(tmp)))
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.beta=np.reshape(tmp,(1,len(tmp)))
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.moving_mean=np.reshape(tmp,(1,len(tmp)))
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.moving_var=np.reshape(tmp,(1,len(tmp)))
                n_features=len(tmp)
                layer.Velgamma=np.zeros((1,n_features))
                layer.Velbeta=np.zeros((1,n_features))
                layer.RMSPgamma=np.zeros((1,n_features))
                layer.RMSPbeta=np.zeros((1,n_features))
                layer.Gradgamma=np.zeros((1,n_features))
                layer.Gradbeta=np.zeros((1,n_features))
                layer.small_constant=float(f.readline().strip())
                line=f.readline().strip()
            elif line=='dropout_layer':
                layer=dropout_layer(1,0.5)
                layer.prob=float(f.readline().strip())
                layer.nw=float(f.readline().strip())
                line=f.readline().strip()
            elif line=='onehead_attention':
                layer=onehead_attention(1,1,1,1)
                masksize=int(f.readline().strip())
                layer.mask=np.array([[0.0 if x<=y else np.NINF for x in range(masksize)] for y in range(masksize)])
                layer.K,layer.V,layer.Q=unbiased_fc_layer(1,1,1,linear_activation,linear_derivative),unbiased_fc_layer(1,1,1,linear_activation,linear_derivative),unbiased_fc_layer(1,1,1,linear_activation,linear_derivative)
                layer.K.W=np.array([[float(x) for x in neu.split(',')] for neu in f.readline().strip().split(';')])
                layer.V.W=np.array([[float(x) for x in neu.split(',')] for neu in f.readline().strip().split(';')])
                layer.Q.W=np.array([[float(x) for x in neu.split(',')] for neu in f.readline().strip().split(';')])
                n_neurons,n_weights=len(layer.K.W),len(layer.K.W[0])
                layer.K.Velw=np.zeros((n_neurons,n_weights))
                layer.K.RMSPw=np.zeros((n_neurons,n_weights))
                layer.K.Gradw=np.zeros((n_neurons,n_weights)) 
                layer.V.Velw=np.zeros((n_neurons,n_weights))
                layer.V.RMSPw=np.zeros((n_neurons,n_weights))
                layer.V.Gradw=np.zeros((n_neurons,n_weights))
                layer.Q.Velw=np.zeros((n_neurons,n_weights))
                layer.Q.RMSPw=np.zeros((n_neurons,n_weights))
                layer.Q.Gradw=np.zeros((n_neurons,n_weights))
                layer.normfact=n_neurons
                line=f.readline().strip()
            elif line=='layernorm_layer':
                layer=batchnorm_layer(1)
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.gamma=np.reshape(tmp,(1,len(tmp)))
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.beta=np.reshape(tmp,(1,len(tmp)))
                layer.small_constant=float(f.readline().strip())
                n_features=len(tmp)
                layer.Velgamma=np.zeros((1,n_features))
                layer.Velbeta=np.zeros((1,n_features))
                layer.RMSPgamma=np.zeros((1,n_features))
                layer.RMSPbeta=np.zeros((1,n_features))
                layer.Gradgamma=np.zeros((1,n_features))
                layer.Gradbeta=np.zeros((1,n_features))
                line=f.readline().strip()
            elif line=='multihead_attention':
                layer=multihead_attention(1,1,1,1,1,1)
                layer.load_weights(f)
                line=f.readline().strip()
            elif line=='embedding_layer':
                layer=embedding_layer(1,1,1)
                layer.table=np.array([[float(x) for x in neu.split(',')] for neu in f.readline().strip().split(';')])
                n_elements,emb_size=len(layer.table),len(layer.table[0])
                layer.Grad=np.zeros((n_elements,emb_size))
                layer.Vel=np.zeros((n_elements,emb_size))
                layer.RMSP=np.zeros((n_elements,emb_size))
                layer.count=np.zeros((n_elements,),dtype=int) 
                line=f.readline().strip() 
            elif line=='learnable_positional_embedding':
                layer=learnable_positional_embedding(1,1,1)
                layer.table=np.array([[float(x) for x in neu.split(',')] for neu in f.readline().strip().split(';')])
                n_elements,emb_size=len(layer.table),len(layer.table[0])
                layer.Grad=np.zeros((n_elements,emb_size))
                layer.Vel=np.zeros((n_elements,emb_size))
                layer.RMSP=np.zeros((n_elements,emb_size))
                line=f.readline().strip()
            elif line=='residual_block':
                layer=residual_block([])
                layer.load_weights(f)
            self.components+=[layer] 
            ncomp-=1
        return line

class unbiased_fc_layer:
    def __init__(self,n_weights,n_neurons,intorno,act_func,act_derivative):
        self.W=np.random.uniform(-intorno,intorno,(n_neurons,n_weights))
        self.Velw=np.zeros((n_neurons,n_weights))
        self.RMSPw=np.zeros((n_neurons,n_weights))
        self.act_func=act_func
        self.act_derivative=act_derivative
        self.Gradw=np.zeros((n_neurons,n_weights))
    
    def activate(self,inps): # inps e' una matrice
        zetas=np.dot(inps,self.W.T)
        return self.act_func(zetas)

    def test_activate(self,inps):
        return self.activate(inps)

    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        self.Velw=mrate*self.Velw+(1-mrate)*self.Gradw # update w vel
        self.RMSPw=rmsprate*self.RMSPw+(1-rmsprate)*self.Gradw**2 # update w rmsp
        self.W+=-1*lrate*((self.Velw/(1-mrate))/(np.sqrt(self.RMSPw/(1-rmsprate))+small_constant)) # update w
        self.Gradw*=0 # set gradients to zero

    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor): # set average_factor to one in sequences learning
        rawgrads=self.act_derivative(acts_of_lay)*act_grad
        self.Gradw+=np.dot(rawgrads.T,inps_to_lay)/average_factor
        #if len(self.Gradb)!=1: # questo if e else e' perche cupy da errore se prova a sommare column vector lungo asse 0. 
        #    self.Gradb+=np.sum(rawgrads,0)/average_factor          il cupy del pc di laura; su kronos non da' errore
        #else:
        #    self.Gradb+=np.sum(rawgrads)/average_factor
        return np.dot(rawgrads,self.W) # return next activation gradients  
         
    def save_weights(self,file_object):
        print('unbiased_fc_layer',file=file_object)
        if self.act_func is relu_activation:
            print('relu_activation',file=file_object)
        elif self.act_func is sig_activation:
            print('sigmoid_activation',file=file_object)
        elif self.act_func is linear_activation:
            print('linear_activation',file=file_object)
        elif self.act_func is tanh_activation:
            print('tanh_activation',file=file_object)
        print(';'.join([','.join([str(x) for x in neuwei]) for neuwei in self.W]),file=file_object)

class softmax_layer:
    def __init__(self):
        pass
    def activate(self,inps): # inps e' una matrice
        exps=np.exp(inps)
        sums=np.sum(exps,1).reshape(len(exps),1)
        return exps/sums
    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor):
        dX = []
        for dy, x in zip(act_grad, inps_to_lay):
            dxi = []
            for dyi, xi in zip(*np.atleast_2d(dy, x)):
                yi = self.activate(xi.reshape(1, -1)).reshape(-1, 1)
                dyidxi = np.diagflat(yi) - yi @ yi.T  # jacobian wrt. input sample xi
                dxi.append(dyi @ dyidxi)
            dX.append(dxi)
        return np.array(dX).reshape(*inps_to_lay.shape)

    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        pass

class positional_encoding:
    def __init__(self):
        self.rawposenc=None
    def activate(self,inps): # inps e' una matrice
        if self.rawposenc is None:
            rawposenc=np.array([[i/10000**(2*j/inps.shape[-1]) for j in range(inps.shape[-1])] for i in range(inps.shape[-2])]) # manca, come ovunque, la batchezza
            rawposenc[:,0::2]=np.sin(rawposenc[:,0::2])
            rawposenc[:,1::2]=np.cos(rawposenc[:,1::2])
            self.rawposenc=rawposenc
        return inps+self.rawposenc
    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor):
        return act_grad
    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        pass

class learnable_positional_embedding:
    def __init__(self,n_elements,emb_size,init):
        """n_elements sarebbe il numero massimo di posizioni, cioe' la max sequence length"""
        self.table=np.random.uniform(-init,init,(n_elements,emb_size))
        self.Grad=np.zeros((n_elements,emb_size))
        self.Vel=np.zeros((n_elements,emb_size))
        self.RMSP=np.zeros((n_elements,emb_size))
    def activate(self,inps):
        return self.table+inps
    def test_activate(self,inps):
        rawpos=np.array([self.table[i] for i in range(len(inps))])
        return rawpos+inps 
    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor):
        self.Grad+=act_grad/average_factor
        # se per caso la sequenza e' piu corta della table, da' errore (broadcast etc).
        # in quel caso mettere un if e aggiungere righe di zero all'activation gradient, ma if len() potrebbe essere time consuming quindi meglio per ora evitare
        return act_grad # passa gli act_grad all'embedding layer
    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        self.Vel=mrate*self.Vel+(1-mrate)*self.Grad
        self.RMSP=rmsprate*self.RMSP+(1-rmsprate)*self.Grad**2
        self.table+=-1*lrate*((self.Vel/(1-mrate))/(np.sqrt(self.RMSP/(1-rmsprate))+small_constant))
        self.Grad*=0
    def save_weights(self,file_object):
        print('learnable_positional_encoding',file=file_object)
        print(';'.join([','.join([str(x) for x in neuwei]) for neuwei in self.table]),file=file_object)

class onehead_attention:
    def __init__(self,n_weights,n_neurons,intorno,masksize): # n_neurons e' la dimensione delle matrici che moltiplicano q, k, o v. Le faccio di uguale dimensione a priori (che ha senso visto che agiscono sulla stessa cosa)
        """masksize e' uguale alla seqlen"""
        self.Q=unbiased_fc_layer(n_weights,n_neurons,intorno,linear_activation,linear_derivative)
        self.K=unbiased_fc_layer(n_weights,n_neurons,intorno,linear_activation,linear_derivative)
        self.V=unbiased_fc_layer(n_weights,n_neurons,intorno,linear_activation,linear_derivative)
        self.softmax=softmax_layer()
        self.normfact=np.sqrt(n_neurons)
        self.mask=np.array([[0.0 if x<=y else np.NINF for x in range(masksize)] for y in range(masksize)])
        self.softweights=None # ricordo per backpropagation
        self.XK=None # ricordo per backpropagation
        self.XV=None # ricordo per backpropagation
        self.XQ=None # ricordo per backpropagation
        self.softmaxinput=None
    def activate(self,inps): # inps e' la sequenza
        XK=self.K.activate(inps)
        self.XK=XK
        XV=self.V.activate(inps)
        self.XV=XV
        XQ=self.Q.activate(inps)
        self.XQ=XQ
        x=np.dot(XQ,XK.T)/self.normfact
        x+=self.mask
        self.softmaxinput=x
        softweights=self.softmax.activate(x)
        self.softweights=softweights
        context=np.dot(softweights,XV)
        return context
    def test_activate(self,inps):
        XK=self.K.activate(inps)
        self.XK=XK
        XV=self.V.activate(inps)
        self.XV=XV
        XQ=self.Q.activate(inps)
        self.XQ=XQ
        x=np.dot(XQ,XK.T)/self.normfact
        x+=self.mask[:len(inps),:len(inps)] #niente maschera in test
        self.softmaxinput=x
        softweights=self.softmax.activate(x)
        self.softweights=softweights
        context=np.dot(softweights,XV)
        return context
    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        self.Q.adam_move_toward_gradients(lrate,mrate,rmsprate,small_constant=small_constant)
        self.K.adam_move_toward_gradients(lrate,mrate,rmsprate,small_constant=small_constant)
        self.V.adam_move_toward_gradients(lrate,mrate,rmsprate,small_constant=small_constant)
    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor):
        softgrad=np.dot(act_grad,self.XV.T)
        unsoftgrad=self.softmax.update_gradients(self.softmaxinput,self.softweights,softgrad,average_factor)
        xqgrad=np.dot(unsoftgrad,self.XK)/self.normfact
        xkgrad=np.dot(unsoftgrad.T,self.XQ)/self.normfact
        xvgrad=np.dot(self.softweights.T,act_grad)
        embgradq=self.Q.update_gradients(inps_to_lay,self.XQ,xqgrad,average_factor)
        embgradk=self.K.update_gradients(inps_to_lay,self.XK,xkgrad,average_factor)
        embgradv=self.V.update_gradients(inps_to_lay,self.XV,xvgrad,average_factor)
        return embgradq+embgradk+embgradv
    def save_weights(self,file_object):
        print('onehead_attention',file=file_object)
        print(len(self.mask),file=file_object)
        print(';'.join([','.join([str(x) for x in neuwei]) for neuwei in self.K.W]),file=file_object)
        print(';'.join([','.join([str(x) for x in neuwei]) for neuwei in self.V.W]),file=file_object)
        print(';'.join([','.join([str(x) for x in neuwei]) for neuwei in self.Q.W]),file=file_object)

class multihead_attention:
    def __init__(self,nheads,finaloutput_nneurons,n_weights,n_neurons,intorno,masksize): # n_neurons e' la dimensione delle matrici che moltiplicano q, k, o v. Le faccio di uguale dimensione a priori (che ha senso visto che agiscono sulla stessa cosa)
        """masksize e' uguale alla seqlen"""
        self.heads=[onehead_attention(n_weights,n_neurons,intorno,masksize) for _ in range(nheads)]
        self.outlay=unbiased_fc_layer(n_neurons*nheads,finaloutput_nneurons,intorno,linear_activation,linear_derivative)
        self.head_size=n_neurons
        self.concatenation=None # per backpropagation dell'outlay
    def activate(self,inps):
        concatenation=np.concatenate([head.activate(inps) for head in self.heads],1) # manca la batchezza
        self.concatenation=concatenation
        return self.outlay.activate(concatenation)
    def test_activate(self,inps):
        concatenation=np.concatenate([head.test_activate(inps) for head in self.heads],1) # manca la batchezza
        self.concatenation=concatenation
        return self.outlay.test_activate(concatenation)
    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        self.outlay.adam_move_toward_gradients(lrate,mrate,rmsprate,small_constant=small_constant)
        for head in self.heads: head.adam_move_toward_gradients(lrate,mrate,rmsprate,small_constant=small_constant)
    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor):
        gradtoconcat=self.outlay.update_gradients(self.concatenation,acts_of_lay,act_grad,average_factor)
        embgrads=np.full_like(inps_to_lay,0.0)
        # i gradienti a ciascuna testa sono i gradienti alla concatenazione divisi ordinatamente
        for i in range(len(self.heads)):
            embgrads+=self.heads[i].update_gradients(inps_to_lay,None,gradtoconcat[:,self.head_size*i:self.head_size*i+self.head_size],1) # manca la batchezza
        return embgrads
    def save_weights(self,file_object):
        print('multihead_attention',file=file_object)
        print(len(self.heads),file=file_object)
        for head in self.heads:
            head.save_weights(file_object)
        print(';'.join([','.join([str(x) for x in neuwei]) for neuwei in self.outlay.W]),file=file_object)
        
    def load_weights(self,infile):
        nheads=int(infile.readline().strip())
        infile.readline().strip() # salto la prima scritta onehead_attention
        self.heads=[]
        while nheads>0:
            layer=onehead_attention(1,1,1,1)
            masksize=int(infile.readline().strip())
            layer.mask=np.array([[0.0 if x<=y else np.NINF for x in range(masksize)] for y in range(masksize)])
            layer.K,layer.V,layer.Q=unbiased_fc_layer(1,1,1,linear_activation,linear_derivative),unbiased_fc_layer(1,1,1,linear_activation,linear_derivative),unbiased_fc_layer(1,1,1,linear_activation,linear_derivative)
            layer.K.W=np.array([[float(x) for x in neu.split(',')] for neu in infile.readline().strip().split(';')])
            layer.V.W=np.array([[float(x) for x in neu.split(',')] for neu in infile.readline().strip().split(';')])
            layer.Q.W=np.array([[float(x) for x in neu.split(',')] for neu in infile.readline().strip().split(';')])
            n_neurons,n_weights=len(layer.K.W),len(layer.K.W[0])
            layer.K.Velw=np.zeros((n_neurons,n_weights))
            layer.K.RMSPw=np.zeros((n_neurons,n_weights))
            layer.K.Gradw=np.zeros((n_neurons,n_weights)) 
            layer.V.Velw=np.zeros((n_neurons,n_weights))
            layer.V.RMSPw=np.zeros((n_neurons,n_weights))
            layer.V.Gradw=np.zeros((n_neurons,n_weights))
            layer.Q.Velw=np.zeros((n_neurons,n_weights))
            layer.Q.RMSPw=np.zeros((n_neurons,n_weights))
            layer.Q.Gradw=np.zeros((n_neurons,n_weights))
            layer.normfact=np.sqrt(n_neurons)
            line=infile.readline().strip()
            nheads-=1
            self.heads.append(layer)
        self.head_size=n_neurons
        layer=unbiased_fc_layer(1,1,1,linear_activation,linear_derivative)
        layer.W=np.array([[float(x) for x in neu.split(',')] for neu in line.split(';')])
        layer.RMSPw=np.zeros((len(layer.W),len(layer.W[0])))
        layer.Velw=np.zeros((len(layer.W),len(layer.W[0])))
        layer.Gradw=np.zeros((len(layer.W),len(layer.W[0]))) 
        self.outlay=layer
         
class embedding_layer:
    def __init__(self,n_elements,emb_size,init):
        self.table=np.random.uniform(-init,init,(n_elements,emb_size))
        self.Grad=np.zeros((n_elements,emb_size))
        self.Vel=np.zeros((n_elements,emb_size))
        self.RMSP=np.zeros((n_elements,emb_size))
        self.count=np.zeros((n_elements,),dtype=int)    
    def activate(self,inps): # inps is a sequence of integers, manca la batchezza
        embeddings=[]
        for i in inps:
            embeddings.append(self.table[i])
            self.count[i]+=1
        return(np.array(embeddings))    
    def adam_move_toward_gradients(self,lrate,mrate,rmsprate,small_constant=1e-7):
        #self.Vel=mrate*self.Vel+(1-mrate)*self.Grad 
        #self.RMSP=rmsprate*self.RMSP+(1-rmsprate)*self.Grad**2 
        #self.table+=-1*lrate*((self.Vel/(1-mrate))/(np.sqrt(self.RMSP/(1-rmsprate))+small_constant))
        #self.Grad*=0 # set gradients to zero
        for i in range(len(self.count)):
            if self.count[i]>0:
                self.Vel[i]=mrate*self.Vel[i]+(1-mrate)*self.Grad[i]
                self.RMSP[i]=rmsprate*self.RMSP[i]+(1-rmsprate)*self.Grad[i]**2
                self.table[i]+=-1*lrate*((self.Vel[i]/(1-mrate))/(np.sqrt(self.RMSP[i]/(1-rmsprate))+small_constant))
                self.Grad[i]*=0
                self.count[i]*=0
    def update_gradients(self,inps_to_lay,acts_of_lay,act_grad,average_factor):
        for i in range(len(inps_to_lay)):
            self.Grad[inps_to_lay[i]]+=act_grad[i]/average_factor     
        # ritorna act grads? non credo esista occasione dove serve
    def save_weights(self,file_object):
        print('embedding_layer',file=file_object)
        print(';'.join([','.join([str(x) for x in neuwei]) for neuwei in self.table]),file=file_object)

class network: # per qualche ragione con numpy il gradiente e' giusto, mentre con cupy viene il triplo. per correggere il gradiente, mettere gradient_correction a 3 in inizializzazione

    def __init__(self,layers,gradient_correction=1,lrate=0.001,mrate=0.9,rmsprate=0.999,softmax=False,cost_function=squared_delta_cost,reg_function=l2_regularization,expected_activations_function=sigmoid_expected_activations,cost_derivative=squared_delta_cost_derivative): # inplay e outlay sono istanze della classe input_output_layer. hidlays è una tupla di istanze della classe hidden_layer
        """gradient correction should be set to 3 if cupy is used"""
        self.architecture=[x for x in layers] # l'architettura e' una lista di classi *_layer
        self.lrate=lrate
        self.mrate=mrate
        self.rmsprate=rmsprate
        self.gradcorr=gradient_correction
        self.cost_function=cost_function
        self.cost_derivative=cost_derivative
        self.reg_function=reg_function
        self.expected_activations_function=expected_activations_function
        self.softmax=True

    def wave_from_inputs(self,inps):
        allinps=[inps]
        for l in range(len(self.architecture)):
            acts=self.architecture[l].activate(inps)
            inps=acts
            allinps.append(inps)
        return allinps # allinps ha sia la last layer activation, sia gli inputs --> len(allinps)=len(self.architecture)+1
    
    def cost_and_backpropagation(self,inps,labels):
        # forward pass
        allacts=self.wave_from_inputs(inps)
        expected_activations=np.zeros((len(labels),len(allacts[-1][0]))) # questa non va bene per batchare sequences
        if len(expected_activations[0])==1:
            for i in range(len(labels)):
                expected_activations[i][0]=float(1-labels[i])
        else:
            for i in range(len(labels)):
                expected_activations[i][labels[i]]=1.0
        if self.softmax: # se softmax, la last layer activations è convertita tramite softmax
            exps=np.exp(allacts[-1])
            somma=np.sum(exps,1)
            last_layer_activations=exps/somma.reshape((len(somma),1)) 
        average_cost=self.cost_function(last_layer_activations,expected_activations)/len(labels)
        # backpropagation
        activation_gradients=self.cost_derivative(last_layer_activations,expected_activations)/self.gradcorr 
        for l in range(len(self.architecture)-1,-1,-1): # allacts[l+1] corrisponde all'attivazione de l-esimo layer, allacts[l] sono gli inps a l-esimo layer, architecture[l] corrisponde a l-esimo layer
            activation_gradients=self.architecture[l].update_gradients(allacts[l],allacts[l+1],activation_gradients,len(labels))        
        return average_cost

    def update_parameters(self):
        for layer in self.architecture:
            layer.adam_move_toward_gradients(self.lrate,self.mrate,self.rmsprate)

    def linear_batch_gd(self,training_data,validation_data=None,batch_size=100,epoch_number=1,quiet=False,lrschedule=tuple(),shiftinps=False,shuffle=True): 
        """lrschedule: (ogniquanteepoche, jump, upperlim, lowerlim, upordown_iniziale, stability)"""
        print('start') if not quiet else None
        inps,labs=training_data
        val_accuracy='None'
        if lrschedule:
            epoch_to_change,change,uplim,lowlim,upordown,stability=lrschedule
        else:
            epoch_to_change=-1 # in this way epoch_to_change will be never 0
        for _ in range(epoch_number):
            ggg=len(labs)
            now=time.time()
            if shuffle: inps,labs=randomizer(training_data)
            tot_epoch_cost=0
            totbatches=0            
            for i in range(0,ggg,batch_size):
                if ggg-i<batch_size:
                    break
                totbatches+=1
                #inputs,labels=np.asarray(inps[i:i+batch_size]),labs[i:i+batch_size] 
                inputs,labels=np.asarray(inps[i:i+batch_size]),np.asarray(labs[i:i+batch_size]) 
                cost=self.cost_and_backpropagation(inputs,labels)
                tot_epoch_cost+=cost
                self.update_parameters()
                print('last_cost =',cost) if not quiet else None
                print('status: epoch=',_+1,'training set done',(i+batch_size)/ggg) if not quiet else None
            if validation_data is not None:
                predicted_labs=self.predict(validation_data[0])
                val_accuracy=accuracy_ontest(validation_data[1],predicted_labs)
            print('epoch=',_+1,'lrate=',self.lrate,'average epoch cost=',tot_epoch_cost/totbatches,'validation accuracy',val_accuracy,'epoch_time=',time.time()-now)
            epoch_to_change-=1 
            if epoch_to_change==0: # update learning rate
                newlr=lrupdown(self,change,uplim,lowlim,upordown) 
                if upordown=='up':
                    if newlr==uplim:
                        upordown='down' 
                        self.lrate=newlr
                        epoch_to_change=stability
                        continue
                self.lrate=newlr
                epoch_to_change=lrschedule[0]
            if shiftinps:
                inps,labs=inps[1:],labs[1:]
                print('shiftcheck:',len(inps),inps[0])
        return cost

    def predict(self,inps):
        for layer in self.architecture:
            if isinstance(layer,batchnorm_layer) or isinstance(layer,dropout_layer) or isinstance(layer,learnable_positional_embedding) or isinstance(layer,residual_block):
                inps=layer.test_activate(inps)
            else:
                inps=layer.activate(inps)
        labels=[]
        for inp in inps:
            if len(inp)>1:
                labels.append(np.argmax(inp))
            else:
                labels.append(0) if inp[0]>=0.5 else labels.append(1)
        return labels

    def save_weights(self,outfile):
        new=open(outfile,'w')
        for layer in self.architecture:
            layer.save_weights(new)
        new.close()
    
    def load_weights(self,infile):
        self.architecture=[]
        f=open(infile)
        line=f.readline().strip()
        while line:
            if line=='fully_connected_layer':
                layer=fully_connected_layer(1,1,1,1,1) # inizializza a caso il layer
                act_fun=f.readline().strip()
                if act_fun=='sigmoid_activation':
                    layer.act_func=sig_activation
                    layer.act_derivative=sigmoid_derivative
                elif act_fun=='relu_activation':
                    layer.act_func=relu_activation
                    layer.act_derivative=relu_derivative
                elif act_fun=='linear_activation':
                    layer.act_func=linear_activation
                    layer.act_derivative=linear_derivative
                weights=f.readline().strip()
                layer.W=np.array([[float(x) for x in neu.split(',')] for neu in weights.split(';')])
                layer.B=np.array([float(x) for x in f.readline().strip().split(';')])
                n_neurons=len(layer.W)
                n_weights=len(layer.W[0])
                layer.Velw=np.zeros((n_neurons,n_weights))
                layer.RMSPw=np.zeros((n_neurons,n_weights))
                layer.Velb=np.zeros((n_neurons,))
                layer.RMSPb=np.zeros((n_neurons,))
                layer.Gradw=np.zeros((n_neurons,n_weights)) 
                layer.Gradb=np.zeros((n_neurons,))
                line=f.readline().strip()
            elif line=='batchnorm_layer':
                layer=batchnorm_layer(1)
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.gamma=np.reshape(tmp,(1,len(tmp)))
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.beta=np.reshape(tmp,(1,len(tmp)))
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.moving_mean=np.reshape(tmp,(1,len(tmp)))
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.moving_var=np.reshape(tmp,(1,len(tmp)))
                n_features=len(tmp)
                layer.Velgamma=np.zeros((1,n_features))
                layer.Velbeta=np.zeros((1,n_features))
                layer.RMSPgamma=np.zeros((1,n_features))
                layer.RMSPbeta=np.zeros((1,n_features))
                layer.Gradgamma=np.zeros((1,n_features))
                layer.Gradbeta=np.zeros((1,n_features))
                layer.small_constant=float(f.readline().strip())
                line=f.readline().strip()
            elif line=='dropout_layer':
                layer=dropout_layer(1,0.5)
                layer.prob=float(f.readline().strip())
                layer.nw=float(f.readline().strip())
                line=f.readline().strip()
            elif line=='onehead_attention':
                layer=onehead_attention(1,1,1,1)
                masksize=int(f.readline().strip())
                layer.mask=np.array([[0.0 if x<=y else np.NINF for x in range(masksize)] for y in range(masksize)])
                layer.K,layer.V,layer.Q=unbiased_fc_layer(1,1,1,linear_activation,linear_derivative),unbiased_fc_layer(1,1,1,linear_activation,linear_derivative),unbiased_fc_layer(1,1,1,linear_activation,linear_derivative)
                layer.K.W=np.array([[float(x) for x in neu.split(',')] for neu in f.readline().strip().split(';')])
                layer.V.W=np.array([[float(x) for x in neu.split(',')] for neu in f.readline().strip().split(';')])
                layer.Q.W=np.array([[float(x) for x in neu.split(',')] for neu in f.readline().strip().split(';')])
                n_neurons,n_weights=len(layer.K.W),len(layer.K.W[0])
                layer.K.Velw=np.zeros((n_neurons,n_weights))
                layer.K.RMSPw=np.zeros((n_neurons,n_weights))
                layer.K.Gradw=np.zeros((n_neurons,n_weights)) 
                layer.V.Velw=np.zeros((n_neurons,n_weights))
                layer.V.RMSPw=np.zeros((n_neurons,n_weights))
                layer.V.Gradw=np.zeros((n_neurons,n_weights))
                layer.Q.Velw=np.zeros((n_neurons,n_weights))
                layer.Q.RMSPw=np.zeros((n_neurons,n_weights))
                layer.Q.Gradw=np.zeros((n_neurons,n_weights))
                layer.normfact=n_neurons
                line=f.readline().strip()
            elif line=='layernorm_layer':
                layer=layernorm_layer(1)
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.gamma=np.reshape(tmp,(1,len(tmp)))
                tmp=np.array([float(x) for x in f.readline().strip().split(';')])
                layer.beta=np.reshape(tmp,(1,len(tmp)))
                layer.small_constant=float(f.readline().strip())
                n_features=len(tmp)
                layer.Velgamma=np.zeros((1,n_features))
                layer.Velbeta=np.zeros((1,n_features))
                layer.RMSPgamma=np.zeros((1,n_features))
                layer.RMSPbeta=np.zeros((1,n_features))
                layer.Gradgamma=np.zeros((1,n_features))
                layer.Gradbeta=np.zeros((1,n_features))
                line=f.readline().strip()
            elif line=='multihead_attention':
                layer=multihead_attention(1,1,1,1,1,1)
                layer.load_weights(f)
                line=f.readline().strip()
            elif line=='embedding_layer':
                layer=embedding_layer(1,1,1)
                layer.table=np.array([[float(x) for x in neu.split(',')] for neu in f.readline().strip().split(';')])
                n_elements,emb_size=len(layer.table),len(layer.table[0])
                layer.Grad=np.zeros((n_elements,emb_size))
                layer.Vel=np.zeros((n_elements,emb_size))
                layer.RMSP=np.zeros((n_elements,emb_size))
                layer.count=np.zeros((n_elements,),dtype=int) 
                line=f.readline().strip() 
            elif line=='learnable_positional_encoding':
                layer=learnable_positional_embedding(1,1,1)
                layer.table=np.array([[float(x) for x in neu.split(',')] for neu in f.readline().strip().split(';')])
                n_elements,emb_size=len(layer.table),len(layer.table[0])
                layer.Grad=np.zeros((n_elements,emb_size))
                layer.Vel=np.zeros((n_elements,emb_size))
                layer.RMSP=np.zeros((n_elements,emb_size))
                line=f.readline().strip()
            elif line=='residual_block':
                layer=residual_block([])
                line=layer.load_weights(f)
                
            self.architecture+=[layer]       
        f.close()


########################################################################

### small demonstration ###
"""
l_rate=0.001
momentum=0.9
rmsp=0.99

# any classification problem
train_inputs=None # use 2d array
train_labels=None # use 1d array
test_inputs=None # use 2d array
test_labels=None # use 1d array

input_size=None # write size of the input 
output_classes=None # write the number of classes

batch_size=128
n_epochs=10

# "intorno" is absolute value of the limit of the uniform distribution from which initialize weights.
# n_weights must be either the input_size, or the number of neurons in the previous layer (same for n_features for batchnorm) 
layer_1=fully_connected_layer(n_weights=input_size,n_neurons=150,intorno=he_uniform(input_size),act_func=relu_activation,act_derivative=relu_derivative) 
batchnorm_1=batchnorm_layer(n_features=150)
layer_2=fully_connected_layer(n_weights=150,n_neurons=75,intorno=he_uniform(150),act_func=relu_activation,act_derivative=relu_derivative)
batchnorm_2=batchnorm_layer(n_features=75)
layer_3=fully_connected_layer(n_weights=75,n_neurons=output_classes,intorno=he_uniform(75),act_func=linear_activation,act_derivative=linear_derivative)

model=network(layers=[layer_1,batchnorm_1,layer_2,batchnorm_2,layer_3],
              lrate=l_rate,mrate=momentum,
              rmsprate=rmsp,softmax=True,
              cost_function=cross_entropy_loss,
              cost_derivative=cross_entropy_loss_derivative)

model.linear_batch_gd(training_data=(train_inputs,train_labels), validation_data=None, batch_size=batch_size, epoch_number=n_epochs, quiet=True) 

predicted_labs=model.predict(test_inputs)
accuracy=accuracy_ontest(test_labels,predicted_labs)
print('\naccuracy on test:',accuracy,'\n')
"""