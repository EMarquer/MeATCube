import pandas as pd, numpy as np
import sys, os
try:
    CURRENT_FOLDER = os.path.dirname(__file__) # normal way
except NameError:
    CURRENT_FOLDER = globals()['_dh'][0] # jupyter notebook way
sys.path.append(os.path.join(CURRENT_FOLDER))

import benchmark_one

from IPython.display import display
if __name__ == "__main__":
    summaries = dict()
    sys_argv = sys.argv
        
    for classifier in benchmark_one.CLASSIFIERS:
        for dataset in benchmark_one.DATASETS:
            try:
                sys.argv = sys_argv[:1] + [dataset] + ["-a", list(benchmark_one.CB_LEARNERS.keys())[0]] + ["-c", classifier] + sys_argv[1:]
                args = benchmark_one.parse_args()
                results = benchmark_one.load_results(args=args)
                df = pd.DataFrame.from_records(results["records"])
                best_model = df.sort_values("test_accuracy", ascending=False).groupby("fold").head(1)
                first_model = df.sort_values("CB_size", ascending=False).groupby("fold").head(1)
                best_initial = pd.concat({"best": best_model, "initial": first_model})
                result_summary = best_initial[["CB_size", "test_accuracy"]].groupby(level=0).aggregate(['mean', 'std'])
                result_summary = result_summary.rename({"test_accuracy": "Accuracy", "CB_size": "|CB|"}, axis=1)
                #result_summary = best_initial[["CB_size", "test_accuracy", "ref_accuracy"]].groupby(level=0).aggregate(['mean', 'std'])
                summaries[(args.dataset_name, args.classifier,)] = result_summary
            except FileNotFoundError:
                pass

    s = pd.concat(summaries, axis=0).sort_index(level=0)
    s = s.reset_index(2)
    s = s.pivot(columns='level_2', values=pd.MultiIndex.from_tuples([('Accuracy', 'mean'), ('Accuracy', 'std'), ('|CB|', 'mean'), ('|CB|', 'std')])).swaplevel(2, 1, axis=1).swaplevel(0, 1, axis=1)
    s = s.sort_index(axis=1, level=2, ascending=True).sort_index(axis=1, level=1, ascending=False).sort_index(axis=1, level=0, ascending=False)
    s = s.drop(('initial', '|CB|', 'std'), axis=1)
    s.columns.set_names(None, level=0, inplace=True)
    s[('initial', '|CB|', 'mean')] = s[('initial', '|CB|', 'mean')].apply(lambda x: f"{int(x):d}")
    s[('best', '|CB|', 'mean')] = s[('best', '|CB|', 'mean')].apply(lambda x: f"{x:.1f}") + "$\pm$" +  s[('best', '|CB|', 'std')].apply(lambda x: f"{x:.1f}")
    s = s.drop(('best', '|CB|', 'std'), axis=1)
    for l in ["best", "initial"]:
        s[(l, 'Accuracy', 'mean')] = s[(l, 'Accuracy', 'mean')].apply(lambda x: f"{x*100:.1f}") + "$\pm$" +  s[(l, 'Accuracy', 'std')].apply(lambda x: f"{x*100:.1f}")
        s = s.drop((l, 'Accuracy', 'std'), axis=1)
    s = s.droplevel(2, axis=1)
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
    print(s.replace('%', '\%').replace('_', ' ').replace('|CB|', '$|CB|$').replace("\multirow[c]{6}{*}", "\midrule\multirow{6}{1.75cm}").replace("Haberman S Survival", "Haberman's Survival"))#.style.to_string())