## 

https://www.sciencedirect.com/science/article/abs/pii/S0020025520311580

https://ieeexplore.ieee.org/document/9459514


1. compression rate VS computation time
0. use the same similarity measure for everything, all the time
2. cross validation


>> analytical report

if compression with smith and mc kena, is coat still with a high performance


ordinal as actual ordinal instead of nominal

-----

1. fair comparison of method
2. comparison of time/space complexity
   1. also batching: 1 is better, more degrades more and more

further down the line:
optimize similarity benefits from compression ? i.e. alternate optimize similarity & compression

-----
## 05/07/2024
-----

- Alzaimer data: minimize number of tests, select subset of 15~20 reference patients
~150 patients

-> interface ? "faire varier les coefs, afficher de manière interactive la frontière de decision"


- implementation
  - continuous seems easy to adapt
  - common interface including drawing (and sklearn templates?)
    - energy based model 
  
  - picker: ui
  - b(i)scoat: Coat contimue + boosting (2cas)
  - ct: Coat contimue
  - ctknn: Coat contimue +fonction d'energie de knn



- repair expe real data:
  - scaling?
  - "true" euclidean distance on the numeric data?
  - no tuning
  - only step 0 for kNN



3 ensembles:
- S (base de cas) 
- V (base de validation) 
- Test

faire version naive de continuous coat comme démo de l'implémentation de modèles


## 25/09/2024
- create draft article (LNCS) to organize contributions


## 02/10/2024

## 07/10/2024
### Paper
knowledge containers compensate for each other: bridge the gap by actually doing it

2 graphs to compare the CB+compression algos:
- best perf VS dataset hyperparameters (half-moon)
- percentage of compression at this perf VS dataset hyperparameters (half-moon)

cross-validation to check if select the same region ("stability")
- test relevance of the final CB using the held-out test set

compare expertise of the same case in different methods?

table best-perf VS CB+compression

aim conference: IJCAI ?

OTHER IDEAS:
- random-forest-style approach by sampling on the dimension
- for large dataset, multiple random subsets (to decrease cost due to explosion of $|CB|$) and merge predictions