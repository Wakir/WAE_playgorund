from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from strlearn.streams import StreamGenerator
from strlearn.ensembles import KUE, CDS, UOB, OOB, WAE
from strlearn2.ensembles import WAE as new_WAE, ROSE, OALE
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,  balanced_accuracy_score as bac, precision_score, recall_score, fbeta_score, roc_auc_score
from specificity import specificity
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
                 imbalance=None, gradual=True, n_chunks=100, n_features=40):
        if metrics is None:
            metrics = [accuracy_score]
        if imbalance is None:
            imbalance = [0.50, 0.50]
        self._streams_random_seeds = streams_random_seeds
        self._ensembles = ensembles
        self._metrics = metrics
        self._ensembles_labels = ensembles_labels
        self._proportions = imbalance
        self._gradual_drift = gradual
        self.n_chunks = n_chunks
        self.n_features = n_features
        self._evaluator = TestThenTrain(self._metrics)
        self._scores = np.empty((len(self._streams_random_seeds), len(self._ensembles), self.n_chunks-1, len(self._metrics)))

    def conduct(self, file=None):
        for r_i, r in enumerate(self._streams_random_seeds):

            if self._gradual_drift:
                stream = StreamGenerator(n_chunks=self.n_chunks, chunk_size=500, n_drifts=5, weights=self._proportions,
                                         n_features=self.n_features,
                                         n_informative=self.n_features//6, n_redundant=self.n_features//6,
                                         n_repeated=self.n_features//6, random_state=r, concept_sigmoid_spacing=5)
            else:
                stream = StreamGenerator(n_chunks=self.n_chunks, chunk_size=500, n_drifts=5, weights=self._proportions, n_features=self.n_features,
                                     n_informative=self.n_features//6, n_redundant=self.n_features//6, n_repeated=self.n_features//6, random_state=r)

            self._evaluator.process(stream, self._ensembles)

            self._scores[r_i, :, :, :] = self._evaluator.scores[:, :, :]

        if file is not None:
            print("Saving file " + file)
            np.save(file, self._scores)


def conduct_experiment(streams_random_seeds=None, ensembles=(), ensembles_labels=(), metrics=None,
                 imbalance=None, gradual=True, file="result"):
    e = Experiment(streams_random_seeds, ensembles, ensembles_labels, metrics,imbalance, gradual)
    e.conduct(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='results')
    parser.add_argument("--parallel", dest='parallel', action='store_true')
    parser.set_defaults(parallel=True)
    parser.add_argument("--reference", dest='reference', action='store_true')
    parser.set_defaults(reference=False)
    args = parser.parse_args()

    path = args.path
    try:
        os.mkdir(path)
    except:
        pass

    random_seeds = [4, 13, 42, 44, 666]

    imbalance = [0.05, 0.10, 0.20, 0.30]

    scales = [0.5, 0.7, 1.0]

    chunk_classifiers = [15, 10, 5, 3]

    #base_classifier = GaussianNB()
    #base_classifier = RandomForestClassifier()
    base_classifier = HoeffdingTreeClassifier()
    #base_classifier = SGDClassifier()

    quality_measure = [accuracy_score, bac, roc_auc_score]
    quality_measure_name = ["accuracy", "bac", "roc_auc"]

    #reference_methods = [UOB, OOB,  WAE]
    #reference_methods = [OALE, ROSE, KUE]
    reference_methods = [CDS]

    metrics = [f1_score, g_mean, bac, precision_score, recall_score, specificity, fbeta_score, roc_auc_score]
    #metrics = [recall_score]


    if args.parallel:
        Parallel(n_jobs=-1)(
            delayed(conduct_experiment)(**kwargs)
            for kwargs in
            ({'streams_random_seeds': random_seeds, 'ensembles': (new_WAE(base_estimator=base_classifier, scale=1.0, n_classifiers=10, base_quality_measure= bac , pruning_criterion=quality_measure[q]),),
              'ensembles_labels': (f"new_wae_{str(quality_measure_name[q])}_cl",),
              'metrics': metrics,
              'imbalance': [1 - imb, imb], 'gradual': True,
              'file': os.path.join(path, f'Stream_gradual_drift_{str(100 * imb)}_imbalance_new_wae_{str(quality_measure_name[q])}_cl.npy')}
             for q in range(len(quality_measure)) for imb in imbalance)
             )
        Parallel(n_jobs=-1)(
            delayed(conduct_experiment)(**kwargs)
            for kwargs in
            ({'streams_random_seeds': random_seeds,
              'ensembles': (new_WAE(base_estimator=base_classifier, scale=1.0, n_classifiers=10, base_quality_measure= bac, pruning_criterion=quality_measure[q]),),
              'ensembles_labels': (f"new_wae_{str(quality_measure_name[q])}_cl",),
              'metrics': metrics,
              'imbalance': [1 - imb, imb], 'gradual': False,
              'file': os.path.join(path,
                                   f'Stream_sudden_drift_{str(100 * imb)}_imbalance_new_wae_{str(quality_measure_name[q])}_cl.npy')}
             for q in range(len(quality_measure)) for imb in imbalance)
        )
        if args.reference:
            Parallel(n_jobs=-1)(
                delayed(conduct_experiment)(**kwargs)
                for kwargs in
                ({'streams_random_seeds': random_seeds,
                  'ensembles': (ref(base_estimator=base_classifier),),
                  'ensembles_labels': (ref.__name__,),
                  'metrics': metrics,
                  'imbalance': [1 - imb, imb], 'gradual': True,
                  'file': os.path.join(path,
                                       f'Stream_gradual_drift_{str(100 * imb)}_imbalance_{ref.__name__}.npy')}
                 for ref in reference_methods for imb in imbalance)
            )
            Parallel(n_jobs=-1)(
                delayed(conduct_experiment)(**kwargs)
                for kwargs in
                ({'streams_random_seeds': random_seeds,
                  'ensembles': (ref(base_estimator=base_classifier),),
                  'ensembles_labels': (ref.__name__,),
                  'metrics': metrics,
                  'imbalance': [1 - imb, imb], 'gradual': False,
                  'file': os.path.join(path,
                                       f'Stream_sudden_drift_{str(100 * imb)}_imbalance_{ref.__name__}.npy')}
                 for ref in reference_methods for imb in imbalance))
    else:
        for imb in imbalance:
            if args.reference:
                for ref in reference_methods:
                    conduct_experiment(random_seeds,
                                       (ref(base_estimator=base_classifier),),
                                       (ref.__name__,), metrics, [1 - imb, imb], True,
                                       os.path.join(path,
                                                    f'Stream_gradual_drift_{str(100 * imb)}_imbalance_{ref.__name__}.npy')
                                       )
                    conduct_experiment(random_seeds,
                                       (ref(base_estimator=base_classifier),),
                                       (ref.__name__,), metrics, [1 - imb, imb], False,
                                       os.path.join(path,
                                                    f'Stream_sudden_drift_{str(100 * imb)}_imbalance_{ref.__name__}.npy'))
            """for scale in scales:
                for n in chunk_classifiers:
                    conduct_experiment(random_seeds,
                                       (new_WAE(base_estimator=base_classifier, scale=scale, n_classifiers=n, base_quality_measure=quality_measure),),
                                       (f"new_wae_{str(scale)}_{str(n)}_cl",), metrics, [1 - imb, imb], True,
                                       os.path.join(path,
                                                    f'Stream_gradual_drift_{str(100 * imb)}_imbalance_new_wae_{str(scale)}_{str(n)}_cl.npy')
                                       )
                    conduct_experiment(random_seeds,
                                       (new_WAE(base_estimator=base_classifier, scale=scale, n_classifiers=n, base_quality_measure=quality_measure),),
                                       (f"new_wae_{str(scale)}_{str(n)}_cl",), metrics, [1 - imb, imb], False,
                                       os.path.join(path,
                                                    f'Stream_sudden_drift_{str(100 * imb)}_imbalance_new_wae_{str(scale)}_{str(n)}_cl.npy'))"""



