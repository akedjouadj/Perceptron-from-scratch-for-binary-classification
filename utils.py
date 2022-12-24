import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def initialisation(X):
    """
    function to initialise the perceptron:
    
    Arguments : 
    X : the data, numpy array

    return :
    (W,b): a tuple, the model's parameters, W is the initialized weights vector and b is the initialized bias  
    """
    W=np.random.randn(X.shape[1],1) 
    b=np.random.randn(1)
    return (W,b)

def model(X,W,b):
    """
    function to run the model, compute the operation XW+b
    
    Arguments : 
    X : the data, numpy array
    W : the initialized weights vector, numpy array of 1 column
    b : the initialized bias, scalar

    return :
    A: the vector of probabilities for each data (each row of X) to belong to the class 0 or the class 1  
    """
    Z=X.dot(W)+b
    A=1/(1+np.exp(-Z))
    return A

def log_loss(A, y, epsilon=1e-15):
    """
    function to compute the model loss
    
    Arguments : 
    A : the model output
    y : the real target vector, numpy array of 1 column, with values in {0, 1}
    epsilon : a small positive constant to ensure that you compute the logarithm of a non-null quantity
    
    return : The bernoulli log-likelihood loss between A and y  
    """
    return (1/len(y))*np.sum( -y*np.log(A+epsilon) -(1-y)*np.log(1-A+epsilon) )

def gradients(A,X,y):
    """
    function to compute the model gradients
    
    Arguments : 
    A : the model output
    X : the data, numpy array
    y : the real target vector, numpy array of 1 column, with values in {0, 1}

    return :
    (dW,db): a tuple, the log_loss gradients  
    """
    dW=(1/len(y))*np.dot(X.T,A-y)
    db=(1/len(y))*np.sum(A-y)
    return (dW,db)

def update(dW,db,W,b,learning_rate):
    """
    function to update the model's parmeters W and b with gradient descent algorithm
    
    Arguments :
    dW, db : the gradients output
    W, b : the model parameters
    learning_rate : the model learning rate

    return :
    (W,b): a tuple, the updated model's parameters  
    """
    W=W- learning_rate*dW
    b=b- learning_rate*db
    return (W,b)

def predict(X,W,b, proba = False):
    """
    function to make a prediction with the model
    
    Arguments : 
    X : the data for prediction, numpy array
    W, b : the trained model's parameters
    proba : boolean, if True, the probability will be return instead of the class 

    return : a vector of the prediction, with values in {0,1}
    """
    A=model(X,W,b)
    if proba==False:
        return (A>=0.5)*1
    else:
        return A   


def accuracy(y, y_pred):
    """
    function to evaluate the prediction's accuracy

    Arguments : 
    y : the true target
    y_pred : the predicted target

    return : the accuracy score of the prediction, scalar between 0 and 1
    """
    return np.sum((y==y_pred)*1)/len(y)

def perceptron(X_train, y_train, X_val, y_val, learning_rate=0.001, err=1e-5, max_iter=10000):
    """
    function to train the perceptron

    Arguments : 
    X_tain : the train data, numpy array
    y_train : the train target, numpy array with 1 column, with values in {0,1}
    X_val : the validation data, numpy array
    y_val : the validation target, numpy array with 1 column, with values in {0,1}
    learning_rate : the model learning rate
    err : the threshold under what you consider the gradient of model is null
    max_iter : the maximum iterations of the gradient descent algorithm (important to stop the training when the model not converge)

    return : The model display the train & val loss, the train & val accuracy
    W, b : the trained parameters
    norm_gradL : the model gradient norm  
    """

    W,b=initialisation(X_train)
    A_train=model(X_train,W,b)
    A_val=model(X_val,W,b)
    dW,db=gradients(A_train,X_train,y_train)
    norm_gradL=np.sqrt(np.sum(dW**2+db**2))
    n_iter=1
    train_Loss=[]
    val_Loss=[]
    train_acc=[]
    val_acc=[]
    
    while((norm_gradL>err) & (n_iter<=max_iter)):
        if(n_iter%1000 == 0): 

            # computing the train loss
            train_Loss.append(log_loss(A_train,y_train))
            y_pred_train=predict(X_train,W,b)
            train_acc.append(accuracy(y_train,y_pred_train))
            
            # computing the validation loss
            val_Loss.append(log_loss(A_val,y_val))
            y_pred_val=predict(X_val,W,b)
            val_acc.append(accuracy(y_val,y_pred_val))

            print("train_loss :", log_loss(A_train,y_train),
                  "val_loss :", log_loss(A_val,y_val),
                  "train_acc :", accuracy(y_train,y_pred_train),
                  "val_acc :", accuracy(y_val,y_pred_val),
                  "norm_gradL :", norm_gradL)
        
        # updating the parameters
        W,b=update(dW,db,W,b,learning_rate)
        
        # activation
        A_train=model(X_train,W,b)
        A_val=model(X_val,W,b)
        
        # computing the gradients
        dW,db=gradients(A_train,X_train,y_train)
        norm_gradL=np.sqrt(np.sum(dW**2)+db**2)
        n_iter=n_iter+1

    print("n_iter", n_iter)

    # display the loss and the accuracy
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_Loss, label='train_Loss')
    plt.plot(val_Loss, label='val_Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend()
    plt.show()
    
    return (W,b,norm_gradL) 