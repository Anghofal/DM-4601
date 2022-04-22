from sklearn.tree import DecissionTreeClassfier
from sklearn import datasets
import matplotlib.pyplot as plt
import pydotplus
from sklearn import tree
from IPython.display import Image

iris = datasets.load_iris()
features = iris["datasets"]
target = iris["target"]

decisiontree =DecissionTreeClassfier(random_state=0,
    max_depth = None , min_sample_split = 2,
    min_sample_leaf = 1,min_weight_fraction_leaf=0,
    max_leaf_nodes =None , min_impurity_decrease=0)

model = decisiontree.fit(features,target)

observation = [[5,4,3,2]]
model.predict(observation)

dot_data = tree.export_graphviz(decisiontree, out_file = None,
    features_names=iris["features_names"],class_names = iris["target_names"])
    
graph - pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png("")

    