from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from strlearn.streams import StreamGenerator
import strlearn as sl
from strlearn.ensembles import KUE, CDS, UOB, OOB, WAE
from strlearn2.ensembles import WAE as new_WAE, ROSE, OALE
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,  balanced_accuracy_score as bac, precision_score, recall_score
from imblearn.metrics import geometric_mean_score as g_mean


import argparse

from strlearn2.evaluators import TestThenTrain

import numpy as np
from joblib import Parallel, delayed
import os
from itertools import product

import warnings
warnings.filterwarnings("ignore")


class Experiment:

    def __init__(self, streams_random_seeds=None, ensembles=(), ensembles_labels=(), metrics=None,
                 stream = None, n_features=40):
        if metrics is None:
            metrics = [accuracy_score]
        self._streams_random_seeds = streams_random_seeds
        self._ensembles = ensembles
        self._metrics = metrics
        self._ensembles_labels = ensembles_labels
        self.n_features = n_features
        self.stream = stream
        self._evaluator = TestThenTrain(self._metrics)
        self._scores = np.empty((len(self._streams_random_seeds), len(self._ensembles), self.stream.n_chunks - 1, len(self._metrics)))
        print(np.shape(self._scores))

    def conduct(self, file=None):
        for r_i, r in enumerate(self._streams_random_seeds):

            self._evaluator.process(self.stream, self._ensembles)

            self._scores[r_i, :, :, :] = self._evaluator.scores[:, :, :]

        if file is not None:
            print("Saving file " + file)
            np.save(file, self._scores)


def conduct_experiment(streams_random_seeds=None, ensembles=(), ensembles_labels=(), metrics=None,
                 stream = None, file="result"):
    e = Experiment(streams_random_seeds, ensembles, ensembles_labels, metrics, stream)
    e.conduct(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='results')
    parser.add_argument("--parallel", dest='parallel', action='store_true')
    parser.set_defaults(parallel=True)
    parser.add_argument("--reference", dest='reference', action='store_true')
    parser.set_defaults(reference=True)
    args = parser.parse_args()

    path = args.path
    try:
        os.mkdir(path)
    except:
        pass

    #imbalance = [0.05, 0.10, 0.20, 0.30]

    scales = [0.5, 0.7, 1.0]

    chunk_classifiers = [3, 5, 10, 15]

    #base_classifier = GaussianNB()
    #base_classifier = RandomForestClassifier()
    base_classifier = HoeffdingTreeClassifier()
    #base_classifier = SGDClassifier()

    quality_measure = bac

    reference_methods = [OALE, ROSE, KUE, UOB, OOB,  WAE]
    #reference_methods = [CDS]

    random_seeds = [194, 85, 186, 170, 200]

    metrics = [f1_score, g_mean, bac, precision_score, recall_score]

    names = os.listdir("./ss_streams")
    #print(np.shape(names))
    names.sort()
    counter = 0

for idx, name in enumerate(names):
    stream = sl.streams.NPYParser("./ss_streams/%s" % name, chunk_size=250, n_chunks=random_seeds[idx])


    if args.parallel:
        """Parallel(n_jobs=-1)(
            delayed(conduct_experiment)(**kwargs)
            for kwargs in
            ({'streams_random_seeds': [random_seeds[idx]],
                'ensembles': (new_WAE(base_estimator=base_classifier, scale=scale, n_classifiers=n, base_quality_measure=quality_measure),),
                'ensembles_labels': (f"new_wae_{str(scale)}_{str(n)}_cl",),
                'metrics': metrics,
                'stream': stream,
                'file': os.path.join(path,
                                    f'Stream_{str(name)}_imbalance_new_wae_{str(scale)}_{str(n)}_cl.npy')}
                for scale in scales for n in chunk_classifiers)
        )"""
        if args.reference:
            Parallel(n_jobs=-1)(
                delayed(conduct_experiment)(**kwargs)
                for kwargs in
                ({'streams_random_seeds': [random_seeds[idx]],
                    'ensembles': (ref(base_estimator=base_classifier),),
                    'ensembles_labels': (ref.__name__,),
                    'metrics': metrics,
                    'stream': stream,
                    'file': os.path.join(path,
                                        f'Stream_{str(name)}_imbalance_{ref.__name__}.npy')}
                    for ref in reference_methods))
    else:
        if args.reference:
            for ref in reference_methods:
                conduct_experiment([random_seeds[idx]],
                                    (ref(base_estimator=base_classifier),),
                                    (ref.__name__,), metrics, stream,
                                    os.path.join(path,
                                                f'Stream_sudden_drift_{str(name)}_imbalance_{ref.__name__}.npy'))
        """for scale in scales:
            for n in chunk_classifiers:
                conduct_experiment([random_seeds[idx]],
                                    (new_WAE(base_estimator=base_classifier, scale=scale, n_classifiers=n, base_quality_measure=quality_measure),),
                                    (f"new_wae_{str(scale)}_{str(n)}_cl",), metrics,
                                    os.path.join(path,
                                                f'Stream_sudden_drift_{str(name)}_imbalance_new_wae_{str(scale)}_{str(n)}_cl.npy'))"""



