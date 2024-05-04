## Run algorithms

Commands:
```
python qiga.py
```

## Datasets

__Gset__ and synthetic data are used.

[Gset](https://web.stanford.edu/~yyye/yyye/Gset/) was created by Stanford University.

Take g14.txt (an undirected graph with 800 nodes and 4694 edges) as an example:

800 4694 # #nodes is 800, and #edges is 4694.
 
1 7 1 # node 1 connects with node 7, weight = 1

1 10 1 # node 1 connects node 10,  weight = 1

1 12 1 # node 1 connects node 12, weight = 1


## Solutions 

Solutions will be written to a file xxx.txt in the folder "binary_solutions". It is a binary vector of the 500 nodes and their corresponding set.

## Checking Solutions
To check solutions, run ```check_cut.py```.


## Performance

| Nodes | Seed | L2A Cut | Gurobi Cut | My Cut | Improvement (%) from max(L2A, Gurobi) | My Binary Solution Vector* |
|-------|------|---------|------------|--------|--------------------------|--------------------------|
| 500   | 0    | 1470    | **1473**   | 1472   | -0.068                   | binary_solutions/500_0_solution.txt |
| 500   | 3    | 1458    | 1459       | **1463** | 0.274                  | binary_solutions/500_3_solution.txt |
| 500   | 10   | 1473    | 1462       | **1475** | 0.136                  | binary_solutions/500_10_solution.txt |
| 500   | 11   | 1467    | 1467       | **1470** | 0.204                  | binary_solutions/500_11_solution.txt |
| 500   | 21   | 1464    | **1468**   | **1468** | 0.000                  | binary_solutions/500_21_solution.txt |
| 500   | 22   | 1474    | 1474       | **1476** | 0.136                  | binary_solutions/500_22_solution.txt |
| 500   | 23   | 1469    | **1471**   | **1471** | 0.000                  | binary_solutions/500_23_solution.txt |
| 500   | 24   | 1470    | 1469       | **1473** | 0.204                  | binary_solutions/500_24_solution.txt |
| 500   | 27   | 1464    | 1462       | **1465** | 0.068                  | binary_solutions/500_27_solution.txt |
| 500   | 28   | 1463    | **1465**   | 1464    | -0.068                  | binary_solutions/500_28_solution.txt |
| gset_70 | N/A | **9583** | 9490     | 9343    | -2.504                  | binary_solutions/gset_70_solution.txt |


\* My binary solution vectors are provided under the folder "binary_solutions" as the table formatting is no longer clean when pasting them.

\* They are in the format of text files containing a combination of 1's and 0's with spaces between each 1 or 0.

\* I have provided the paths to the relevant files in the table.

L2A's results are represented as strings. How to transfer the strings into binary results? 

Take data/syn/powerlaw_100_ID0.txt as an example, the result is "4SuqhIaQimYjyk_sX" by L2A, which can be transferred to a binary vector by calling the function str_to_bool in EncoderBase64 in evaluator.py. 



