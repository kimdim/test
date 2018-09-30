# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 09:50:08 2018

@author: liuxin
"""
from sklearn.datasets import load_iris   #skit-learn是机器学习的开源包
from sklearn import tree  
from sklearn.externals.six import StringIO  
import pydotplus  
iris = load_iris()#载入数据集  
clf = tree.DecisionTreeClassifier()#算法模型  
clf = clf.fit(iris.data, iris.target)#模型训练  
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")#写入pdf 

