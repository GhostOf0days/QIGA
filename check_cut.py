def evaluate_cut(solution, edges):
    cut_value = 0
    for u, v, w in edges:
        if solution[u] != solution[v]:
            cut_value += w
    return cut_value

# Read graph file
with open('data/gset/gset_70.txt', 'r') as f:
    n, m = map(int, f.readline().split())
    edges = []
    for line in f:
        u, v, w = map(int, line.split())
        edges.append((u - 1, v - 1, w))


# Read solution vector
with open('result/my_result/gset_70_solution_copy.txt', 'r') as f:
    solution = list(map(int, f.read().split()))

cut_value = evaluate_cut(solution, edges)
print(f"The cut value is: {cut_value}")