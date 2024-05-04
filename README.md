## Run algorithms

Commands:
```
python qiga.py
```

## Datasets

Take g14.txt (an undirected graph with 800 nodes and 4694 edges) as an example:

800 4694 # #nodes is 800, and #edges is 4694.
 
1 7 1 # node 1 connects with node 7, weight = 1

1 10 1 # node 1 connects node 10,  weight = 1

1 12 1 # node 1 connects node 12, weight = 1


## Results 

Results will be written to a file xxx.txt in the folder "result". It is a binary vector of the 500 nodes and their corresponding set.

## Checking Results
To check results, run ```check_cut.py```.


## Performance
In the following experiments, we used GPU during training by default. The best-known results are labed in bold.

The results are stored in the folder "result". Take __Gset__ as an example.

[Gset](https://web.stanford.edu/~yyye/yyye/Gset/) was created by Stanford University. j

| Nodes | Seed | L2A Cut | Gurobi Cut | My Cut | Improvement from max(L2A, Gurobi) (%) | Binary Solution Vector |
|-------|------|---------|------------|--------|--------------------------|--------------------------|
| 500   | 0    | 1470    | **1473**   | 1472   | -0.068                   | 1 1 0 1 0 1 0 1 0 1 0 0 0 0 1 0 0 1 1 1 0 0 1 1 1 1 1 1 0 1 0 0 1 1 0 1 0 1 1 0 0 1 1 0 0 1 1 0 0 1 0 1 0 1 0 0 1 1 0 1 0 0 0 1 1 0 0 0 0 1 1 1 1 1 0 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 0 1 1 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 1 0 0 0 0 1 1 0 0 0 1 0 0 1 0 0 0 1 1 1 1 1 0 1 0 0 0 1 1 0 1 1 1 0 1 1 1 0 0 1 0 1 1 1 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 1 1 1 0 1 1 1 0 1 1 0 1 1 1 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 1 1 1 0 0 1 0 1 0 1 1 1 0 0 0 1 0 0 0 1 0 0 1 1 1 1 0 1 0 1 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 0 1 1 0 0 1 0 0 0 0 1 0 1 1 1 1 0 1 0 1 0 0 0 1 1 1 1 1 1 0 0 0 1 1 0 1 0 0 1 1 0 1 1 1 0 1 1 0 1 0 1 1 0 1 1 0 1 0 1 1 0 0 1 1 1 1 0 0 0 1 1 0 1 1 1 0 1 0 1 0 1 1 0 1 0 1 0 0 1 0 0 1 1 1 0 0 0 1 0 0 1 1 0 1 1 1 0 0 1 0 0 0 1 1 1 1 0 1 1 0 0 0 1 0 0 0 1 1 1 0 0 0 1 0 1 0 0 0 1 1 0 1 0 0 0 0 1 0 1 1 0 1 1 0 1 1 1 0 0 0 1 1 1 1 0 1 0 0 1 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 1 1 0 0 0 0 1 0 1 1 0 0 1 0 0 0 0 1 1 0 1 0 0 0 0 1 0 1 1 1 0 0
| 500   | 3    | 1458    | 1459       | **1463** | 0.274                  |
| 500   | 10   | 1473    | 1462       | **1475** | 0.136                  |
| 500   | 11   | 1467    | 1467       | **1470** | 0.204                  |
| 500   | 21   | 1464    | **1468**   | **1468** | 0.000                  |
| 500   | 22   | 1474    | 1474       | **1476** | 0.136                  |
| 500   | 23   | 1469    | **1471**   | **1471** | 0.000                  |
| 500   | 24   | 1470    | 1469       | **1473** | 0.204                  |
| 500   | 27   | 1464    | 1462       | **1465** | 0.068                  |
| 500   | 28   | 1463    | **1465**   | 1464    | -0.068                  |
| gset_70 | N/A | **9583** | 9490     | 9343    | -2.504                  |

L2A's results are represented as strings. How to transfer the strings into binary results? 

Take data/syn/powerlaw_100_ID0.txt as an example, the result is "4SuqhIaQimYjyk_sX" by L2A, which can be transferred to a binary vector by calling the function str_to_bool in EncoderBase64 in evaluator.py. 



