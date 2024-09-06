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
05/07/2024
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
