import os
import re
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def make_datasets(raws_path, path, sample_n):
    
    def _get_pair(df):
        n = len(df)
        test_n = int(n//5)
        mask = np.array([False]*n)
        mask[np.random.choice(n, test_n, replace=False)] = True
        tr_df, te_df = df[np.invert(mask)], df[mask]
        return tr_df, te_df

    def _transform(keyword, tr_df, te_df):
        cols = [v for v in tr_df.columns[1:] if re.compile(keyword).search(v)]
        if not cols:
            return tr_df, te_df
        tr_xs, te_xs = np.array(tr_df[cols]), np.array(te_df[cols])
        pca = PCA()
        pca.fit(tr_xs)
        tr_xs, te_xs = pca.transform(tr_xs), pca.transform(te_xs)
        tr_df[cols], te_df[cols] = tr_xs, te_xs
        return tr_df, te_df

    def get_dataset(opt, term, idx):
        nonlocal raws_path
        np.random.seed(idx)
        
        filepath = os.path.join(raws_path, 'time%d') % term
        df = pd.read_pickle(filepath)
        
        if opt == 'base':
            p = re.compile(r'minwon|news')
            cols = [v for v in df.columns if not p.search(v)]
        elif opt == 'ours':
            cols = df.columns
        elif opt == 'text':
            p = re.compile(r'minwon|news')
            cols = ['Target']+[v for v in df.columns if p.search(v)]
        df = df[cols]
        
        bools = df['Target'] == 0
        tr0, te0 = _get_pair(df[bools])
        tr1, te1 = _get_pair(df[~bools])
        tr_df, te_df = pd.concat([tr0, tr1]), pd.concat([te0, te1])
        
        p = re.compile(r'Target')
        cols = [v for v in df.columns if not p.search(v)]
        tr_xs, te_xs = np.array(tr_df[cols]), np.array(te_df[cols])
        ss = StandardScaler()
        ss.fit(tr_xs)
        tr_df[cols], te_df[cols] = ss.transform(tr_xs), ss.transform(te_xs)
        
        bools = tr_df['Target'] == 0
        up_weight = sum(bools)//sum(~bools)
        tr_df = pd.concat([tr_df[bools]]+[tr_df[~bools]]*up_weight)
        
        tr_df, te_df = _transform(r'_news_topic', tr_df, te_df)
        tr_df, te_df = _transform(r'_minwon_topic', tr_df, te_df)

        sys.stdout.write('\r%s %s %03d' % (opt, filepath, idx))
        return tr_df, te_df

    opts = ['base', 'ours', 'text']
    terms = [1, 2, 3, 4]
    argses = []
    for opt in opts:
        dir_path = os.path.join(path, '%s') % opt
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
        for term in terms:
            dir1path = os.path.join(dir_path, 'time%d') % term
            if os.path.isdir(dir1path):
                shutil.rmtree(dir1path)
            os.mkdir(dir1path)
            for i in range(sample_n):
                args = (opt, term, i+1)
                argses.append(args)
            
    filepath = os.path.join(path, '%s', 'time%d', 'data%03d')
    for args in argses:
        dataset = get_dataset(*args)
        pd.to_pickle(dataset, filepath % args)
    print()


if __name__ == "__main__":

    raws_path = os.path.join('data', 'raws')
    make_datasets(raws_path, 'data', 50)
