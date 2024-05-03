def evaluate_cut(solution, edges):
    cut_value = 0
    for u, v, w in edges:
        if solution[u-1] != solution[v-1]:
            cut_value += w
    return cut_value

# Read graph file
with open('data/syn/powerlaw_500_ID28.txt', 'r') as f:
    n, m = map(int, f.readline().split())
    edges = []
    for line in f:
        u, v, w = map(int, line.split())
        edges.append((u, v, w))

# Read solution vector
with open('result/result_syn_gurobi_QUBO/powerlaw_500_ID28_3600.txt', 'r') as f:
    lines = f.readlines()
    solution = [0] * n
    for line in lines:
        if line.startswith('//'):
            continue
        node, part = map(int, line.split())
        solution[node-1] = part

cut_value = evaluate_cut(solution, edges)
print(f"The cut value is: {cut_value}")