from sklearn.naive_bayes import GaussianNB
from strlearn.ensembles.WAE import WAE
from strlearn.streams.StreamGenerator import StreamGenerator
from strlearn.evaluators.TestThenTrain import TestThenTrain
from strlearn.metrics import balanced_accuracy_score

stream = StreamGenerator()
clf = WAE(GaussianNB(), post_pruning=True)
ttt = TestThenTrain(
metrics=(balanced_accuracy_score))
ttt.process(stream, clf)
print(ttt.scores)