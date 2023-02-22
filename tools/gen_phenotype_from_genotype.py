import numpy as np
from tools.activation_functions import normalize

def genotype_to_phenotype(net, size_params):
    """ 
    
    """
    # Gets the inputs for 
    x_inputs = np.zeros(size_params)
    y_inputs = np.zeros(size_params)
    z_inputs = np.zeros(size_params)
    
    for x in range(size_params[0]):
            for y in range(size_params[1]):
                for z in range(size_params[2]):
                    x_inputs[x, y, z] = x
                    y_inputs[x, y, z] = y
                    z_inputs[x, y, z] = z

    x_inputs = normalize(x_inputs)
    y_inputs = normalize(y_inputs)
    z_inputs = normalize(z_inputs)

    #Creates the d input array, calculating the distance each point is away from the centre
    d_inputs = normalize(np.power(np.power(x_inputs, 2) + np.power(y_inputs, 2) + np.power(z_inputs, 2), 0.5))

    #Creates the b input array, which is just a numpy array of ones
    b_inputs = np.ones(size_params)

    #Sets all inputs and flattens them into 1D arrays
    x_inputs = x_inputs.flatten()
    y_inputs = y_inputs.flatten()
    z_inputs = z_inputs.flatten()
    d_inputs = d_inputs.flatten()
    b_inputs = b_inputs.flatten()
    
    inputs = list(zip(x_inputs, y_inputs, z_inputs, d_inputs, b_inputs))
    
    body_size = 1
    for dim in size_params:
        body_size *= dim
    body = np.zeros(body_size)
        
    for n, input in enumerate(inputs):
        output = net.activate(input) # Gets output from activating CPPN
        presence = output[0]
        material = output[1]

        if presence <= 0.2: #Checks if presence output is less than 0.2
            body[n] = 0 #If so there is no material in the location
        elif material < 0.5: #Checks if material output is less than 0.5 
            body[n] = 1 #If so there is skin in the location
        else:
            body[n] = 2 #Else there is a cardiac cell in the location   
    
    return body 