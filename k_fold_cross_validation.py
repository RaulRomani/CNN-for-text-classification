import numpy as np





                                        # x_train
def k_fold_cross_validation(K, l2_penalty, data, output_name):
    
    sum_error = 0
    for k in xrange(0,K):        #segment
        #spliting the data
        validation_set = extractSegment(data=data, k=K, segment=k)
        training_set   = extractRemider(data=data, k=K, segment=k)
        
        # fit the model
        model = graphlab.linear_regression.create(training_set, 
                                                  target = output_name, 
                                                  l2_penalty = l2_penalty,
                                                  validation_set = None, verbose=False) 
                
        residual = validation_set[output_name] - model.predict(validation_set)
        RSS = (residual*residual).sum()
        #validation_error = RSS/len(validation_set)
        sum_error += RSS
    CV = sum_error/K # a.k.a. MSE (mean square error)
    return CV
        

#leave k out cross validation    
def lko(data, input_name, output_name, deg, l2_penalty_values,num_folds): # data = ['X1' , 'Y']
    # Create polynomial features
    data = polynomial_features(data[input_name], data[output_name], output_name, deg) # data = ['X1' , 'Y', 'X2', ... , 'Xn']        
    
    # for each value of l2_penalty, fit a model for each fold and compute average MSE
    l2_penalty_mse = []
    min_mse = None
    best_l2_penalty = None
    for l2_penalty in l2_penalty_values:          
        next_mse = k_fold_cross_validation(num_folds, l2_penalty, data, output_name)        
        l2_penalty_mse.append(next_mse)
        if min_mse is None or next_mse < min_mse:
            min_mse = next_mse
            best_l2_penalty = l2_penalty    




# l2_penalty=1e5
# print(l2_penalty)


l2_penalty_values = np.logspace(1, 7, num=13)
print ("l2_penalty = ", l2_penalty_values[3])

# l2_penalty_mse, best_l2_penalty = lko(data, 'sqft_living', 'price', 15, l2_penalty_values,10)

