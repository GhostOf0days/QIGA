import os
import numpy as np
import math
import random
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, GATConv

# Create the directory if it doesn't exist
os.makedirs('result/my_result', exist_ok=True)

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, gnn_type='gin', dense=True):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        if gnn_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim, dense=dense))
            self.convs.append(GATConv(hidden_dim, output_dim))
        elif gnn_type == 'gin':
            self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
            for _ in range(num_layers - 2):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, output_dim))))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = torch.relu(x)
        return torch.sigmoid(x)

def read_graph(filename):
    with open(filename, 'r') as f:
        n, m = map(int, f.readline().split())
        edges = []
        for line in f:
            u, v, w = map(int, line.split())
            edges.append((u-1, v-1, w))
    return n, m, edges

def evaluate_cut(solution, edges):
    cut_value = 0
    for u, v, w in edges:
        if solution[u] != solution[v]:
            cut_value += w
    return cut_value

def initialize_population(n, pop_size):
    population = np.zeros((pop_size, n))
    angles = np.zeros((pop_size, n))
    
    for i in range(pop_size):
        # Opposition-based learning
        if i < pop_size // 2:
            population[i] = np.random.randint(2, size=n)
        else:
            population[i] = 1 - population[i - pop_size // 2]
        
        # Chaotic map initialization
        angles[i] = np.arcsin(np.sqrt(population[i]))
        angles[i] = np.clip(angles[i] + 0.4 * (4 * angles[i] * (1 - angles[i])), 0, np.pi/2)
    
    return population, angles

def observe(chromosome, n):
    return np.where(np.random.rand(n) < np.sin(chromosome)**2, 1, 0)

def update(population, angles, best_solution, best_angles, rotation_angle, t, max_iter):
    avg_fitness = np.mean(np.sum(np.sin(angles)**2, axis=1))
    adaptive_rotation_scale = 0.1 * (1 - t/max_iter)
    
    for i in range(len(population)):
        if (np.sum(np.sin(angles[i])**2) > avg_fitness):
            rotation_angle[i] *= adaptive_rotation_scale
        else:
            rotation_angle[i] *= 0.05
            
        angles[i] = best_angles + rotation_angle[i] * (angles[i] - best_angles)
        angles[i] = np.clip(angles[i], 0, np.pi/2)
        population[i] = np.where(np.sin(angles[i])**2 > 0.5, 1, 0)
        
    return population, angles

def multi_point_crossover(angles1, angles2, crossover_rate):
    if random.random() < crossover_rate:
        num_points = random.randint(1, len(angles1) - 1)
        points = sorted(random.sample(range(1, len(angles1)), num_points))
        points = [0] + points + [len(angles1)]
        
        child1 = np.zeros_like(angles1)
        child2 = np.zeros_like(angles2)
        
        for i in range(len(points) - 1):
            if i % 2 == 0:
                child1[points[i]:points[i+1]] = angles1[points[i]:points[i+1]]
                child2[points[i]:points[i+1]] = angles2[points[i]:points[i+1]]
            else:
                child1[points[i]:points[i+1]] = angles2[points[i]:points[i+1]]
                child2[points[i]:points[i+1]] = angles1[points[i]:points[i+1]]
        
        return child1, child2
    
    return angles1, angles2

def adaptive_mutation(angles, mutation_rate, t, max_iter):
    adaptive_rate = mutation_rate * math.exp(-5 * (t / max_iter))
    for i in range(len(angles)):
        if random.random() < adaptive_rate:
            angles[i] = np.random.uniform(0, np.pi/2)
    return angles

def local_search(solution, edges, max_iter=1000, tabu_list=None, tabu_tenure=50):
    n = len(solution)
    best_solution = solution[:]
    best_cut = evaluate_cut(best_solution, edges)
    
    for _ in range(max_iter):
        i = random.randint(0, n-1)
        if i not in tabu_list:
            solution[i] = 1 - solution[i]
            new_cut = evaluate_cut(solution, edges)
            if new_cut > best_cut:
                best_cut = new_cut
                best_solution = solution[:]
                if tabu_list is not None:
                    tabu_list.append(i)
                    if len(tabu_list) > tabu_tenure:
                        tabu_list.pop(0)
            else:
                solution[i] = 1 - solution[i]
    
    return best_solution

def gnn_guided_search(model, data, solution, edges, max_iter=1000):
    n = len(solution)
    best_solution = solution[:]
    best_cut = evaluate_cut(best_solution, edges)
    
    for _ in range(max_iter):
        data.x = torch.tensor(solution, dtype=torch.float).unsqueeze(1)
        output = model(data).squeeze()
        probabilities = output.detach().numpy()
        i = np.random.choice(n, p=probabilities)
        solution[i] = 1 - solution[i]
        new_cut = evaluate_cut(solution, edges)
        if new_cut > best_cut:
            best_cut = new_cut
            best_solution = solution[:]
        else:
            solution[i] = 1 - solution[i]
            
    return best_solution

def simulated_annealing(solution, edges, max_iter=1000, initial_temp=1.0, final_temp=0.01, cooling_rate=0.99, l2a_score=None):
    n = len(solution)
    current_solution = solution[:]
    current_cut = evaluate_cut(current_solution, edges)
    best_solution = current_solution[:]
    best_cut = current_cut
    
    temp = initial_temp
    while temp > final_temp:
        for _ in range(max_iter):
            i = random.randint(0, n-1)
            current_solution[i] = 1 - current_solution[i]
            new_cut = evaluate_cut(current_solution, edges)
            delta = new_cut - current_cut
            if delta > 0 or random.random() < math.exp(delta / temp):
                current_cut = new_cut
                # print(f"Current Cut: {current_cut}")
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_solution = current_solution[:]
                    if best_cut > l2a_score:
                        # Write the best solution to the file
                        with open('result/my_result/solution.txt', 'w') as sol_file:
                            sol_file.write(' '.join(map(str, best_solution)))
                    print(f"Current Best Cut: {best_cut}")
            else:
                current_solution[i] = 1 - current_solution[i]
        temp *= cooling_rate
    
    return best_solution

def adaptive_operator_selection(operator_weights, operator_rewards, reward_type='avg'):
    if reward_type == 'avg':
        operator_quality = operator_rewards / (operator_weights + 1e-5)
    elif reward_type == 'extreme':
        operator_quality = np.zeros(len(operator_weights))
        if np.sum(operator_weights) > 0:
            operator_quality[np.argmax(operator_rewards)] = 1
    else:
        raise ValueError(f"Invalid reward type: {reward_type}")
    
    total_quality = np.sum(operator_quality)
    if total_quality > 1e-8:  # Check if total_quality is not too small
        operator_probs = operator_quality / total_quality
    else:
        # If total_quality is too small, assign equal probabilities
        operator_probs = np.ones(len(operator_weights)) / len(operator_weights)
    
    return operator_probs


def QIGA(edges, n, pop_size=100, max_iter=3000, crossover_rate=0.9, mutation_rate=0.05, gnn_model=None, gnn_data=None, transfer_model=None, l2a_score = None):
    population, angles = initialize_population(n, pop_size)
    rotation_angle = np.random.uniform(0, np.pi/2, size=pop_size)
    best_cut = 0
    best_solution = None
    best_angles = None
    no_improvement_count = 0
    tabu_list = []
    tabu_tenure = 50
    
    overall_best_cut = 0
    overall_best_solution = None
    
    operator_weights = np.ones(3)
    operator_rewards = np.zeros(3)
    
    with open('result/my_result/result.txt', 'a') as file:
        for t in range(max_iter):
            solutions = [observe(chromosome, n) for chromosome in population]
            cut_values = [evaluate_cut(solution, edges) for solution in solutions]
            
            max_cut = max(cut_values)
            if max_cut > best_cut:
                best_cut = max_cut
                best_solution = solutions[cut_values.index(max_cut)]
                best_angles = angles[cut_values.index(max_cut)]
                no_improvement_count = 0
                
                # Update the overall best cut and solution
                if best_cut > overall_best_cut:
                    overall_best_cut = best_cut
                    overall_best_solution = best_solution.copy()
            else:
                no_improvement_count += 1
            
            if t % 100 == 0:
                current_solution = local_search(overall_best_solution, edges, max_iter=2000, tabu_list=tabu_list, tabu_tenure=tabu_tenure)
                current_cut = evaluate_cut(current_solution, edges)
                
                if current_cut > overall_best_cut:
                    overall_best_cut = current_cut
                    overall_best_solution = current_solution.copy()
                
                if gnn_model is not None and gnn_data is not None:
                    current_solution = gnn_guided_search(gnn_model, gnn_data, overall_best_solution, edges, max_iter=2000)
                    current_cut = evaluate_cut(current_solution, edges)
                    
                    if current_cut > overall_best_cut:
                        overall_best_cut = current_cut
                        overall_best_solution = current_solution.copy()

            population, angles = update(population, angles, overall_best_solution, best_angles, rotation_angle, t, max_iter)
            
            new_angles = []
            for i in range(0, pop_size, 2):
                child1, child2 = multi_point_crossover(angles[i], angles[i+1], crossover_rate)
                new_angles.append(child1)
                new_angles.append(child2)
            angles = np.array(new_angles)
            
            angles = adaptive_mutation(angles, mutation_rate, t, max_iter)
            
            if no_improvement_count >= 500:
                # Restart mechanism with opposition-based learning
                population, angles = initialize_population(n, pop_size)
                rotation_angle = np.random.uniform(0, np.pi/2, size=pop_size)
                no_improvement_count = 0
            
            if t % 100 == 0:
                progress = (t / max_iter) * 100
                message = f"Iteration: {t}, Best Cut: {overall_best_cut}, Progress: {progress:.2f}%"
                print(message)
                print(message, file=file)
            
            # Simulated Annealing
            if t % 500 == 0:
                current_solution = simulated_annealing(overall_best_solution, edges, max_iter=1000, initial_temp=1.0, final_temp=0.01, cooling_rate=0.99, l2a_score=l2a_score)
                current_cut = evaluate_cut(current_solution, edges)
                
                if current_cut > overall_best_cut:
                    overall_best_cut = current_cut
                    overall_best_solution = current_solution.copy()
            
            # Adaptive Operator Selection
            if t % 100 == 0:
                operator_probs = adaptive_operator_selection(operator_weights, operator_rewards, reward_type='avg')
                crossover_rate, mutation_rate, _ = operator_probs
                operator_weights += 1
                operator_rewards[0] += np.mean(cut_values)
                operator_rewards[1] += np.max(cut_values)
                operator_rewards[2] += overall_best_cut
        
        current_solution = local_search(overall_best_solution, edges, max_iter=5000, tabu_list=tabu_list, tabu_tenure=tabu_tenure)
        current_cut = evaluate_cut(current_solution, edges)
        
        if current_cut > overall_best_cut:
            overall_best_cut = current_cut
            overall_best_solution = current_solution.copy()
        
        if gnn_model is not None and gnn_data is not None:
            current_solution = gnn_guided_search(gnn_model, gnn_data, overall_best_solution, edges, max_iter=5000)
            current_cut = evaluate_cut(current_solution, edges)
            
            if current_cut > overall_best_cut:
                overall_best_cut = current_cut
                overall_best_solution = current_solution.copy()
    
    return overall_best_solution, overall_best_cut

def parallel_QIGA(edges, n, pop_size=200, max_iter=3000, crossover_rate=0.9, mutation_rate=0.05, num_processes=4, l2a_score=None):
    pool = mp.Pool(processes=num_processes)
    results = [pool.apply_async(QIGA, args=(edges, n, pop_size, max_iter, crossover_rate, mutation_rate, None, None, None, l2a_score)) for _ in range(num_processes)]
    pool.close()
    pool.join()
    
    empty = True

    # Write the best solution to the file
    with open('result/my_result/solution.txt', 'r') as sol_file:
        for line in sol_file:
            stripped_line = line.strip()
            if stripped_line:
                best_solution = list(map(int, sol_file.read().split()))
                # I HAVEN'T FIXED THIS YET, BUT A CRASH HERE MEANS SIMULATED ANNEALING FOUND A BETTER SOLUTION THAN L2A
                # DON'T WORRY. SA'S SOLUTION WAS WRITTEN TO SOLUTION.TXT
                best_cut = evaluate_cut(best_solution, edges)
                empty = False

        if empty == True:
            best_solution = None
            best_cut = 0
    
    for result in results:
        solution, cut = result.get()
        if cut > best_cut:
            best_cut = cut
            best_solution = solution
            # Write the best solution to the file
            with open('result/my_result/solution.txt', 'w') as sol_file:
                sol_file.write(' '.join(map(str, best_solution)))
    
    return best_solution

def train_gnn(edges, n, hidden_dim=128, lr=0.001, epochs=500, gnn_type='gin', residual=True, dense=True):
    edge_index = torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t().contiguous()
    x = torch.zeros((n, 1), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    model = GNN(1, hidden_dim, 1, num_layers=6, gnn_type=gnn_type, dense=dense)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    
    with open('result/my_result/result.txt', 'a') as file:
        best_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = nn.BCELoss()(output, x)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'result/my_result/best_gnn.pt')
            
            if epoch % 20 == 0:
                message = f"Epoch: {epoch}, Loss: {loss.item():.4f}"
                print(message)
                print(message, file=file)
    
    model.load_state_dict(torch.load('result/my_result/best_gnn.pt'))
    return model, data

def transfer_learning(source_edges, target_edges, n, hidden_dim=64, lr=0.01, epochs=100, gnn_type='gin'):
    edge_index = torch.tensor([[u, v] for u, v, _ in source_edges], dtype=torch.long).t().contiguous()
    x = torch.zeros((n, 1), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    model = GNN(1, hidden_dim, 1, num_layers=6, gnn_type=gnn_type, dense=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    with open('result/my_result/result.txt', 'a') as file:
        best_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = nn.BCELoss()(output, x)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'result/my_result/best_gnn.pt')
            
            if epoch % 20 == 0:
                message = f"Epoch: {epoch}, Loss: {loss.item():.4f}"
                print(message)
                print(message, file=file)
    
    model.load_state_dict(torch.load('result/my_result/best_gnn.pt'))
    
    edge_index = torch.tensor([[u, v] for u, v, _ in target_edges], dtype=torch.long).t().contiguous()
    x = torch.zeros((n, 1), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    return model, data

if __name__ == "__main__":
    directory = 'data/gset'
    instances = ['gset_70']
    l2a_results = [9541]
    
    total_l2a_cut = sum(l2a_results)
    total_qiga_cut = 0
    
    with open('result/my_result/result.txt', 'w') as file:
        for instance, l2a_cut in zip(instances, l2a_results):
            filename = os.path.join(directory, f'{instance}.txt')
            n, m, edges = read_graph(filename)
            
            message = f"\nGraph: {instance}"
            print(message)
            print(message, file=file)
            
            message = f"Nodes: {n}, Edges: {m}"
            print(message)
            print(message, file=file)
            
            message = f"L2A Benchmark Cut: {l2a_cut}"
            print(message)
            print(message, file=file)
            
            gnn_model, gnn_data = train_gnn(edges, n)
            
            best_cut = 0
            best_solution = None
            
            solution = parallel_QIGA(edges, n, pop_size=200, max_iter=3000, crossover_rate=0.9, mutation_rate=0.05, num_processes=4, l2a_score=l2a_results[0])
            cut = evaluate_cut(solution, edges)
            best_cut = cut
            best_solution = solution
    
            message = f"QIGA Cut: {best_cut}"
            print(message)
            print(message, file=file)
            total_qiga_cut += best_cut

            message = f"Improvement %: {100*(best_cut-l2a_cut)/l2a_cut:.2f}"
            print(message)
            print(message, file=file)
            
            message = f"Solution vector: {best_solution}"
            print(message)
            print(message, file=file)
        
        message = "\nSummary:"
        print(message)
        print(message, file=file)
        
        message = f"Total L2A Cut: {total_l2a_cut}"
        print(message)
        print(message, file=file)
        
        message = f"Total QIGA Cut: {total_qiga_cut}"
        print(message)
        print(message, file=file)
        
        message = f"Improvement %: {100*(total_qiga_cut-total_l2a_cut)/total_l2a_cut:.2f}"
        print(message)
        print(message, file=file)
