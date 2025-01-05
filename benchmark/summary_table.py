import pandas as pd, numpy as np
import sys, os
try:
    CURRENT_FOLDER = os.path.dirname(__file__) # normal way
except NameError:
    CURRENT_FOLDER = globals()['_dh'][0] # jupyter notebook way
sys.path.append(os.path.join(CURRENT_FOLDER))

import benchmark_one

from IPython.display import display

INCREMENTAL_ALGOS = {"CNNR", "CkNNR"}
if __name__ == "__main__":
    summaries = dict()
    sys_argv = sys.argv

    cb_learners = set()
    for algo in benchmark_one.CB_LEARNERS:
        for classifier in benchmark_one.CLASSIFIERS:  
            for dataset in benchmark_one.DATASETS:
                try:
                    sys.argv = sys_argv[:1] + [dataset] + ["-a", algo] + ["-c", classifier] + sys_argv[1:]
                    args = benchmark_one.parse_args()
                    results = benchmark_one.load_results(args=args)
                    df = pd.DataFrame.from_records(results["records"])
                    best_model = df.sort_values("test_accuracy", ascending=False).groupby("fold").head(1)
                    first_model = df.sort_values("CB_size", ascending=algo in INCREMENTAL_ALGOS).groupby("fold").head(1)
                    max_model = df.sort_values("CB_size", ascending=False).groupby("fold").head(1)
                    #best_model["compression_rate"] = 1. - (best_model["CB_size"].astype(float) / results["args"]["size_S"])
                    best_model["compression_rate"] = best_model["CB_size"].astype(float) / results["args"]["size_S"]
                    best_initial = pd.concat({"best": best_model, "initial": first_model})
                    result_summary = best_initial[["CB_size", "compression_rate", "test_accuracy"]].groupby(level=0).aggregate(['mean', 'std'])
                    result_summary = result_summary.rename({"test_accuracy": "Accuracy", "CB_size": "|CB|"}, axis=1)
                    #result_summary = best_initial[["CB_size", "test_accuracy", "ref_accuracy"]].groupby(level=0).aggregate(['mean', 'std'])
                    summaries[(args.dataset_name, args.classifier, args.algo,)] = result_summary

                    cb_learners.add(algo)
                except (FileNotFoundError, ValueError):
                    pass

    s = pd.concat(summaries, axis=0).sort_index(level=0)
    print(s)
    s = s.reset_index(3)
    s = s.reset_index(2)
    s = s.pivot(columns=['level_2', 'level_3'], 
                values=pd.MultiIndex.from_tuples([('Accuracy', 'mean'), ('Accuracy', 'std'), ('|CB|', 'mean'), ('|CB|', 'std'), ('compression_rate', 'mean'), ('compression_rate', 'std')]))
    s = s.swaplevel(3, 1, axis=1).swaplevel(0, 2, axis=1)
    s = s.sort_index(axis=1, level=2, ascending=True).sort_index(axis=1, level=1, ascending=False).sort_index(axis=1, level=0, ascending=False)
    s.columns.set_names(None, level=0, inplace=True)
    s.columns.set_names(None, level=1, inplace=True)
    for algo in cb_learners:
        s = s.drop((algo, 'initial', '|CB|', 'std'), axis=1)
        s[(algo, 'initial', '|CB|', 'mean')] = s[(algo, 'initial', '|CB|', 'mean')].apply(lambda x: f"{int(x):d}")
        s = s.drop((algo, 'initial', 'compression_rate', 'mean'), axis=1)
        s = s.drop((algo, 'initial', 'compression_rate', 'std'), axis=1)

        s[(algo, 'best', 'compression_rate', 'mean')] = s[(algo, 'best', 'compression_rate', 'mean')].apply(lambda x: f"{x*100:.1f}") + "$\pm$" +  s[(algo, 'best', 'compression_rate', 'std')].apply(lambda x: f"{x*100:.1f}")
        s[(algo, 'best', '|CB|', 'mean')] = s[(algo, 'best', '|CB|', 'mean')].apply(lambda x: f"{x:.1f}") + "$\pm$" +  s[(algo, 'best', '|CB|', 'std')].apply(lambda x: f"{x:.1f}")
        s[(algo, 'best', '|CB|', 'mean')] = s[(algo, 'best', '|CB|', 'mean')] + " (" +  s[(algo, 'best', 'compression_rate', 'mean')] + ")"
        s = s.drop((algo, 'best', '|CB|', 'std'), axis=1)
        s = s.drop((algo, 'best', 'compression_rate', 'std'), axis=1)
        s = s.drop((algo, 'best', 'compression_rate', 'mean'), axis=1)

        s.columns = pd.MultiIndex.from_tuples(s.set_axis(s.columns.values, axis=1)
                                       .rename(columns={(algo, 'best', '|CB|', 'mean'): (algo, 'best', '|CB| (% comp.)', 'mean')}))
        for l in ["best", "initial"]:
            s[(algo, l, 'Accuracy', 'mean')] = s[(algo, l, 'Accuracy', 'mean')].apply(lambda x: f"{x*100:.1f}") + "$\pm$" +  s[(algo, l, 'Accuracy', 'std')].apply(lambda x: f"{x*100:.1f}")
            s = s.drop((algo, l, 'Accuracy', 'std'), axis=1)
    s = s.droplevel(3, axis=1)
    print(s)
    s= s.style
    #s = s.format('{:.2f}', "|CB|")
    #s = s.format( '{:.2%}', ["Accuracy"]).to_latex()
    #s = pd.concat(summaries, axis=0).sort_index(level=0).style.format('{:.2f}', "CB_size").format( '{:.2%}', ["test_accuracy", "ref_accuracy"]).to_latex()
    s = s.set_table_styles([
        {'selector': 'th',
        'props': [
            ('border-style', 'solid'),
            ('border-color', 'Red'),
            ('vertical-align','top')
        ]
        }]
    )
    s = s.to_latex()
    
    s = s.replace('%', '\%')
    s = s.replace('_', ' ')
    s = s.replace('|CB|', '$|CB|$')
    s = s.replace("\multirow[c]{6}{*}", "\midrule\multirow{6}{1.75cm}")
    s = s.replace("\multicolumn{2}{r}", "\multicolumn{2}{c}")
    s = s.replace("\multicolumn{4}{r}", "\multicolumn{4}{c}")
    cols = ''.join(["l"]*4*len(benchmark_one.CB_LEARNERS))
    newcols = ''.join(["|cccc"]*len(benchmark_one.CB_LEARNERS))
    s = s.replace("begin{tabular}{ll" + cols + "}", "begin{tabular}{ll" + newcols + "}")
    s = s.replace("Haberman S Survival", "Haberman's Survival")
    
    print(s)#.style.to_string())