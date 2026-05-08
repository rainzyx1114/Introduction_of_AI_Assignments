import numpy as np
import modelLogisticRegression as LR
import modelTree as Tree
import modelRandomForest as Forest
import modelSoftmaxRegression as SR
import modelMultiLayerPerceptron as MLP
import YourTraining as YT
from scipy import ndimage
import pickle

class NullModel:

    def __init__(self):
        pass

    def __call__(self, figure):
        return 0


class LRModel:
    def __init__(self) -> None:
        with open(LR.save_path, "rb") as f:
            self.weight, self.bias = pickle.load(f)

    def __call__(self, figure):
        pred = figure @self.weight + self.bias
        return 0 if pred > 0 else 1

class TreeModel:
    def __init__(self) -> None:
        with open(Tree.save_path, "rb") as f:
            self.root = pickle.load(f)
    
    def __call__(self, figure):
        return Tree.inferTree(self.root, Tree.discretize(figure.flatten()))


class ForestModel:
    def __init__(self) -> None:
        with open(Forest.save_path, "rb") as f:
            self.roots = pickle.load(f)
    
    def __call__(self, figure):
        return Forest.infertrees(self.roots, Forest.discretize(figure.flatten()))


class SRModel:
    def __init__(self) -> None:
        with open(SR.save_path, "rb") as f:
            graph = pickle.load(f)
        self.graph = graph
        self.graph.eval()

    def __call__(self, figure):
        self.graph.flush()
        pred = self.graph.forward(figure, removelossnode=True)[-1]
        return np.argmax(pred, axis=-1)
    
class MLPModel:
    def __init__(self) -> None:
        with open(MLP.save_path, "rb") as f:
            graph = pickle.load(f)
        self.graph = graph
        self.graph.eval()

    def __call__(self, figure):
        self.graph.flush()
        pred = self.graph.forward(figure, removelossnode=True)[-1]
        return np.argmax(pred, axis=-1)

class MyModel:
    def __init__(self) -> None:
        with open(YT.save_path, "rb") as f:
            graph = pickle.load(f)
        self.graph = graph
        self.graph.eval()

    def predict(self, graph, figure):
        graph.eval() 
        probs_sum = None
        angles = [0, +3, -3]
        for angle in angles:
            X_aug = []
            for img in figure:
                img2d = img.reshape(28, 28)
                if angle != 0:
                    img_rot = ndimage.rotate(img2d, angle, reshape=False, order=1)
                else:
                    img_rot = img2d
                X_aug.append(img_rot.reshape(-1))
            X_aug = np.array(X_aug)
            graph.flush()
            logits = graph.forward(X_aug, removelossnode=True)[-1]
            if probs_sum is None:
                probs_sum = np.exp(logits)
            else:
                probs_sum += np.exp(logits)
        logits_avg = np.log(probs_sum / len(angles))
        return logits_avg

    def __call__(self, figure):
        self.graph.flush()
        # pred = self.predict(self.graph, figure)
        pred = self.graph.forward(figure, removelossnode=True)[-1]
        return np.argmax(pred, axis=-1)

modeldict = {
    "Null": NullModel,
    "LR": LRModel,
    "Tree": TreeModel,
    "Forest": ForestModel,
    "SR": SRModel,
    "MLP": MLPModel,
    "Your": MyModel
}

