import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    plt.show(features)

class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.
       
        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations


    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        itns = 0
        weights = np.zeros(3)
        weights[0] = 1
        y = np.ones(features.shape[0])
        classif = np.ones(features.shape[0])
        features = np.column_stack((np.ones(features.shape[0]), features))
        prev_weights = np.ones(3)
        while (not np.allclose(prev_weights, weights)) or itns < self.max_iterations:
            prev_weights = weights
            for i in range(features.shape[0]):
                y[i] = np.dot(weights, features[i,:])
                if y[i] > 0:
                    classif[i] = 1
                elif y[i] == 0:
                    classif[i] = 0
                else:
                    classif[i] = -1
                if targets[i] != classif[i]:
                    weights = prev_weights + features[i, :] * targets[i]
            itns += 1
        self.weights = weights
        print(weights)

    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        features = np.column_stack((np.ones(features.shape[0]), features))
        predictions = np.array([])
        multiplied = np.array([])
        for i in range(features.shape[0]):
            multiplied = np.append(multiplied, np.dot(features[i,:], self.weights))
        for i in range(len(multiplied)):
            if multiplied[i] < 0:
                predictions = np.append(predictions, -1)
            else:
                predictions = np.append(predictions, 1)
        return predictions


    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        raise NotImplementedError()
