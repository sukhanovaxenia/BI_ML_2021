import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
           
        Clues:
        iterate, firstly, over train set and inside it - over test set;
           then subtract from each train all test values;
           take module by got difference
           sum resulting values to gat the overall distance of pictures
        """
        
        dist_mtrx = np.zeros((X.shape[0], self.train_X.shape[0]))

        for j in range(self.train_X.shape[0] - 1):
            for i in range(X.shape[0] - 1):
                dist_elem = self.train_X[j] - X[i]
                module = abs(dist_elem)
                dist_mtrx[i,j] = sum(module)
        
        return dist_mtrx
        #pass


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
           
        Clues:
        subtract - alternative to 'difference': train - test;
            abs - module;
            sum - need to set axis = 1 (otherwise will sum by rows);
            np is used 'cause otherwise will not be able to apply broadcasting.
            
        """

        dist_mtrx2 = np.zeros((X.shape[0], self.train_X.shape[0]))
        for i in range(0,X.shape[0] - 1):
            dist_mtrx2[i, :] = np.sum(np.abs(np.subtract(X[i],self.train_X)), axis = 1) #using broadcasting
        
        return dist_mtrx2
        #pass


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
           
        Clues:
        To generate pairwise distance estimation only by one line we have to solve dimension problem;
        The basic option - add one more special exis to one of the set;
        Extra dimension will provide an opportunity to re-scale resulting matrix ~ broadcasting;
        axis = -1 set by which axis we ought to sum values (herein - the 3d one, set as -1)
        
        """
        
        dist_mtrx3 = np.sum(np.abs(self.train_X - X[:, None]), axis = -1)
       
        return dist_mtrx3
        #pass


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        
        Clues:
        predictions - empty array with set type of data = int;
        necessary to avoid conversion to float while addition of values
        
        iterate over testing samples and distances,
                sorting distances and search indicies among:
                    a) the one from test dist equal to the first k distances
                                               and get the most frequent one;
                    b) for each dist in array, extracting only the most frequent
        
        """
        n_train = distances.shape[1]
        n_test = distances.shape[0]
        predictions = np.empty(n_test, dtype = int)
        classes = []
        k = self.k
        #iterate over testing samples
        for i in range(n_test):
            #get distances for the i-sample
            test_D = distances[i,:]
            #sort distances
            sorted_D = list(sorted(test_D, key = lambda x: x)) #lambda tells to sort by the 1st arg in array - dist 
            #if k is set - take first k elements from sorted distances
            if k is not None:
                kD_sort = sorted_D[:k]
                #iter over first min distances
                for min_D in range(len(kD_sort)):
                    #get indicies of dist from default set equal to the min one
                    idx = list(test_D).index(kD_sort[min_D])
                    #extract min class from train set and append to list
                    class_min = self.train_y.astype(int)[idx]
                    classes.append(class_min)
                    #try found the only most frequent class
                    try:
                        sample_class = max(classes, key = classes.count)
                    #if more than one frequent - get min
                    except ValueError:
                        sample_class = class_min
                predictions[i] = sample_class
            else:
                for min_D in range(len(sorted_D)):
                    idx = list(test_D).index(sorted_D[min_D])
                    class_min = self.train_y.astype(int)[idx]
                    classes.append(class_min)
                    try:
                        sample_class = max(classes, key = classes.count)
                    except ValueError:
                        sample_class = class_min
                predictions[i] = sample_class
        return predictions
        


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        Clues:
        predictions - empty array with set type of data = int;
        necessary to avoid conversion to float while addition of values
        
        iterate over testing samples and distances,
                sorting distances and search indicies among:
                    a) the one from test dist equal to the first k distances
                                               and get the most frequent one;
                    b) for each dist in array, extracting only the most frequent
        
        """
        n_train = distances.shape[1]
        n_test = distances.shape[0]
        predictions = np.empty(n_test, dtype = int)
        classes = []
        k = self.k
        #iterate over testing samples
        for i in range(n_test):
            #get distances for the i-sample
            test_D = distances[i,:]
            #sort distances
            sorted_D = list(sorted(test_D, key = lambda x: x)) #lambda tells to sort by the 1st arg in array - dist 
            #if k is set - take first k elements from sorted distances
            if k is not None:
                kD_sort = sorted_D[:k]
                #iter over first min distances
                for min_D in range(len(kD_sort)):
                    #get indicies of dist from default set equal to the min one
                    idx = list(test_D).index(kD_sort[min_D])
                    #extract min class from train set and append to list
                    class_min = self.train_y.astype(int)[idx]
                    classes.append(class_min)
                    #try found the only most frequent class
                    try:
                        sample_class = max(classes, key = classes.count)
                    #if more than one frequent - get min
                    except ValueError:
                        sample_class = class_min
                predictions[i] = sample_class
            else:
                for min_D in range(len(sorted_D)):
                    idx = list(test_D).index(sorted_D[min_D])
                    class_min = self.train_y.astype(int)[idx]
                    classes.append(class_min)
                    try:
                        sample_class = max(classes, key = classes.count)
                    except ValueError:
                        sample_class = class_min
                predictions[i] = sample_class
        return predictions
        
