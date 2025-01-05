
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier

from logging import warning, info

def euclidean_sim(x1, x2):
    return np.exp(-euclidean(x1, x2))

def class_equality_sim(y1, y2):
    return np.equal(y1, y2).all().astype(float)

def cnnr(config):
    return CNN(config).cnnr()

class CNN():
    """Main class for Condensed Nearest Neighbor variants."""
    def __init__(self, config) -> None:
        self.config = config

    def cnnr(self):
        """Condensed Nearest Neighbor Rule, original 1NN version, as of Hart, 1967."""
        learning_time = 0.0
        start = time.time()
        config = self.config
        (X,y) = (config.X,config.y)

        clf = KNeighborsClassifier(n_neighbors=1)

        folds = config.generated_train_validation_test_sets()
        for (i, fold) in enumerate(folds):
            (X_S, y_S, X_V, y_V, X_T, y_T) = (X[fold['S']], y[fold['S']], X[fold['V']], y[fold['V']], X[fold['T']], y[fold['T']])
            info(f'cnnr({config.dataset_name}) -- fold {i+1}/{len(folds)} |S|={len(X_S)} |V|={len(X_V)} |T|={len(X_T)}')
            store = (np.array([X_S[0]]), np.array([y_S[0]]))
            clf.fit(*store)
            
            grabbag = np.array(list(range(1, len(X_S))))
            removed_cases = []
            changes = True
            while changes:
                changes = False
                misclassified = []

                for j in grabbag:
                    (sj, rj) = (X_S[j], y_S[j])
                    y_pred = clf.predict(sj[None, :])[0]
                    if y_pred != rj:
                        if j not in misclassified:
                            misclassified.append(j)
                            store = np.vstack((store[0], sj)), np.concatenate((store[1], [rj]))
                            clf.fit(*store)
                        changes = True
                for j in misclassified:
                    if j not in removed_cases:
                        removed_cases.append(j)
                    grabbag = np.delete(grabbag, np.where(grabbag == j))

            end = time.time()
            learning_time += end - start

            # first remove the cases given by cnnr, then add random cases
            remaining_cases = np.delete(np.arange(len(X_S)),removed_cases)
            np.random.shuffle(remaining_cases)
            removals = np.concatenate((np.array(removed_cases),remaining_cases))
            
            (acc_V,acc_T,fit_time,eval_time) = ([],[],[],[])
            for j in range(len(X_S)):
                fit_start = time.time()
                clf.fit(np.delete(X_S,removals[:j],axis=0),np.delete(y_S,removals[:j],axis=0))
                fit_end = time.time()
                fit_time.append(fit_end - fit_start)
                pred = clf.predict(X_S)
                acc_V.append(np.mean(pred == y_S))
                eval_start = time.time()
                pred = clf.predict(X_T)
                eval_end = time.time()
                eval_time.append(eval_end - eval_start)
                acc_T.append(np.mean(pred == y_T))
                
            should_include = np.zeros(len(X_S))
            best_j = len(acc_V) -1 - np.array(acc_V)[::-1].argmax()
            should_include[best_j+1:]=1.0
            
            df_i = pd.DataFrame(data={'fold':i+1,
                                    'acc_V':acc_V,
                                    'acc_T':acc_T,
                                    'learn_time':fit_time,
                                    'pred_time':eval_time,
                                    'should_include':should_include
                                    })
            result.add_fold(df_i)
        result.dump()
        result.save_CB(np.delete(X_S,removed_cases,axis=0), np.delete(y_S,removed_cases,axis=0))
        return result