import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    kernel=np.flipud(np.fliplr(kernel)) # flips kernel 
    for i in range (Hi):    
        for j in range (Wi):        #iterates through input 
            total=0                 
            for k in range (Hk):    
                for l in range (Wk):        #same for output
                    
                    x=i+k-Hk//2
                    y=j+l-Wk//2             #position of the image indices for the convolution
                    if 0<=x<Hi and 0<=y<Wi:
                        total+=image[x,y]*kernel[k,l] #does the convolution

                    if x < 0 and y<0:
                        total+=0        #if value out of the indices (edges) makes its contrubution 0
            out[i, j] = total       #writes the convoluted values 
            
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    pass
    out=np.zeros((H+2*pad_height, W+2*pad_width))  #creates a zero matrix of size Height_Input+padding_size_up+padding_size_down
    out[pad_height:H+pad_height,pad_width:W+pad_width]=image #centers the image 
                                                    
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    kernel_fliped=np.flip(kernel)
    padded_image=zero_pad(image, Hk//2,Wk//2)
    for i in range (Hi):
        for j in range (Wi):
            image_piece=padded_image[i:i+Hk,j:j+Wk] #takes the part of the image to which kernel is convoluted to 

            out[i,j]=np.sum(image_piece*kernel_fliped)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pass
    g_prime=np.flip(g)
    out=conv_fast(f,g_prime)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pass
    g_mean=np.mean(g)
    g_no_mean=g-g_mean
    out=cross_correlation(f,g_no_mean)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pass
    Hi,Wi=f.shape
    Hk,Wk=g.shape
    out=np.zeros((Hi,Wi))


    kernel_mean=np.mean(g)
    kernel_dev=np.std(g)

    padded_image=zero_pad(f,Hk//2,Wk//2)
    for i in range (Hi):
        for j in range (Wi):
            image_piece=padded_image[i:i+Hk,j:j+Wk]

            piece_mean=np.mean(image_piece)
            piece_dev=np.std(image_piece)
            
            numerator=np.sum((image_piece-piece_mean)*(g-kernel_mean))
            denominator=piece_dev*kernel_dev*Hk*Wk
            out[i,j]=numerator//denominator

    ### END YOUR CODE

    return out
