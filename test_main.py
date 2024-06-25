from sklearn.tree import DecisionTreeClassifier
from strlearn2.ensembles.WAE import WAE
from strlearn2.streams.StreamGenerator import StreamGenerator
from strlearn2.evaluators.TestThenTrain import TestThenTrain
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,  balanced_accuracy_score as bac, precision_score, recall_score
from strlearn2.metrics import balanced_accuracy_score

weights = [0.7, 0.3]
stream = StreamGenerator(weights = weights)
clf = WAE(DecisionTreeClassifier(), post_pruning=True, base_quality_measure=balanced_accuracy_score)
ttt = TestThenTrain(
metrics=(balanced_accuracy_score))
ttt.process(stream, clf)
print(ttt.scores)