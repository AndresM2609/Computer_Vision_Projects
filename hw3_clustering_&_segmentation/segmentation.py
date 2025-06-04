import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float


### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        pass
        #all point in the original set of data
        for i,point in enumerate(features):   
                distance_to_each_centroid=[[np.linalg.norm(point-centroid) ]for centroid in centers]
                cluster_assingn=np.argmin(distance_to_each_centroid)
                assignments[i]=cluster_assingn

        new_centers=np.zeros_like(centers) 

        for i in range (k): # k was the number of clusters so through each cluster

            #extract all points that where assigned to cluster i
            points_in_cluster=features[assignments==i]  
             #if there are point on the cluster
            if len(points_in_cluster)>0: 
                #does mean for the new center
                new_centers[i]=points_in_cluster.mean(axis=0) 
            # if there were no points on the cluster
            else:
                #its original centroid stays the same
                new_centers[i]=centers[i]       



        if np.allclose(centers, new_centers):   
            break                               
                                        
        
        centers=new_centers
        ### END YOUR CODE

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        pass
        #expands the features so that every element on it can be paired with everx element of the centers 
        distances=np.linalg.norm(features[:, np.newaxis, :]-centers,axis=2)
        #looks on each row(each row has the distance from a point to all the centroids) for each row
        assignments=np.argmin(distances, axis=1)
        
        summed_points=np.zeros((k,D))
        np.add.at(summed_points,assignments,features)
        counts = np.bincount(assignments, minlength=k).reshape(k, 1)
        new_centers=np.where(counts!= 0, summed_points/counts, centers)

        if np.allclose(centers,new_centers):
            break

        centers=new_centers
        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        ### YOUR CODE HERE
        pass
        distances=[]
        pairs=[]
        for i in range (len(centers)):
            for j in range (i+1,len(centers)):
                distance=np.linalg.norm(centers[i]-centers[j])
                distances.append(distance)
                pairs.append((i,j))
        
        distances=np.array(distances)

        min_distance_index=np.argmin(distances)
        closest_pair=pairs[min_distance_index]

        i,j =closest_pair
        #merges the points in the j cluster in the i cluster 
        assignments[assignments==j]=i 
        #gets the 'values' of the elements in the merged cluster
        points_in_merged_cluster=features[assignments==i]
        new_center=points_in_merged_cluster.mean(axis=0)

        centers[i]=new_center
        #deletes the center of the point that merged to another cluster
        centers=np.delete(centers,j,axis=0)
        assignments[assignments>j]-=1

        n_clusters-=1


                
        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    pass
    features= img.reshape(H*W,C)
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    pass
    color=color.reshape(H*W,C)              #does the color thing where the color is spread through the full row
    x_cords,y_cords=np.mgrid[0:H,0:W]       #creates 2 2D arrays one with the x(row) values repeated on all coloums and the same for the coloum values on the other 2D array
    x_cords=x_cords.flatten().reshape(H*W,1) #makes the 2D array in a 1D with all the data on a row or coloum 
    y_cords=y_cords.flatten().reshape(H*W,1)

    features=np.hstack((color,x_cords,y_cords)) #this now gives the C,H*W but here the H*W is the position expresed on just 1 dimention

    mean=features.mean(axis=0) #gets the mean 
    std=features.std(axis=0)#gets the deviation 
    features=(features-mean)/std #normalizes 
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    H, W, C = img.shape
    img_normalized = img / 255.0
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    x_coords = x_coords.reshape(-1, 1)
    y_coords = y_coords.reshape(-1, 1)

    img_flat = img_normalized.reshape(H * W, C)
    features = np.concatenate([img_flat, x_coords, y_coords], axis=1)
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    pass
    matching_pixels=np.sum(mask_gt==mask)
    total_pixel=mask_gt.size
    accuracy=matching_pixels/total_pixel

    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
