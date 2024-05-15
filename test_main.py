from sklearn.tree import DecisionTreeClassifier
from strlearn2.ensembles.WAE import WAE
from strlearn2.streams.StreamGenerator import StreamGenerator
from strlearn2.evaluators.TestThenTrain import TestThenTrain
from strlearn2.metrics import balanced_accuracy_score

weights = [0.7, 0.3]
stream = StreamGenerator(weights = weights)
clf = WAE(DecisionTreeClassifier(), post_pruning=True)
ttt = TestThenTrain(
metrics=(balanced_accuracy_score))
ttt.process(stream, clf)
print(ttt.scores)