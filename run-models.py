
import re
import os
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tools.rnn import RecurrentNeuralNetwork

from sklearn.metrics import recall_score, precision_score, f1_score


def get_scores_map(name, opt, term):

    def _get_pair(df):
        ys = np.array(df['Target'])
        df = df.drop('Target', axis=1)
        xs = np.array(df)
        return xs, ys

    def get_dataset(filepath):
        tr_df, te_df = pd.read_pickle(filepath)
        tr_xs, tr_ys = _get_pair(tr_df)
        te_xs, te_ys = _get_pair(te_df)
        return tr_xs, tr_ys, te_xs, te_ys

    def get_lr():
        model = LogisticRegression(
            solver='newton-cg',
            random_state=0
        )
        return model

    def get_rf():
        model = RandomForestClassifier(
            n_estimators=100,
            max_features=None,
            random_state=0
        )
        return model

    def get_nn():
        nonlocal opt, term
        model = RecurrentNeuralNetwork( 
            opt, term,
            epoch_n=150,
            learning_rate=0.01,
            regularization_rate=0.002
        )
        return model

    func_map = {'nn': get_nn, 'rf': get_rf, 'lr': get_lr}
    scores_map = {'recall': [], 'precision': [], 'f1-score': []}
    dir_path = os.path.join('data', '%s', 'time%d') % (opt, term)
    filepaths = [os.path.join(dir_path, v) for v in os.listdir(dir_path)
                 if re.compile(r'data\d{3}').match(v)]

    for idx, filepath in enumerate(filepaths):
        tr_xs, tr_ys, te_xs, te_ys = get_dataset(filepath)
        model = func_map[name]()

        if name == "nn":
            preds = model.predict(tr_xs, tr_ys, te_xs, te_ys, idx)
        elif name in {"rf", "lr"}:
            model.fit(tr_xs, tr_ys)
            preds = model.predict(te_xs)

        if all(preds == 0):
            scores_map['precision'].append(0)
            scores_map['f1-score'].append(0)
        else:
            scores_map['precision'].append(precision_score(te_ys, preds))
            scores_map['f1-score'].append(f1_score(te_ys, preds))
        scores_map['recall'].append(recall_score(te_ys, preds))
        
        sys.stdout.write('\r%s' % filepath)

    return scores_map


if __name__ == "__main__":

    names = ['nn', 'rf', 'lr']
    opts = ['base', 'ours', 'text']
    terms = [1, 2, 3, 4]
    total_score_map = {}

    for name in names:
        print(name)
        for opt in opts:
            print(opt)
            for term in terms:
                scores_map = get_scores_map(name, opt, term)
                print("\r", {k: "%0.04f(%0.04f)" % (np.mean(scores_map[k]),
                                                    np.std(scores_map[k]))
                            for k in scores_map})
                total_score_map[name, opt, term] = scores_map

    pd.to_pickle(total_score_map, os.path.join('data', 'total_score_map'))
