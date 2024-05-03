import os
import numpy as np
import math
import random
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GINConv, SAGEConv, SuperGATConv

# Create the directory if it doesn't exist
os.makedirs('result/my_solution_2', exist_ok=True)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdvancedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, gnn_type='supergat'):
        super(AdvancedGNN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        if gnn_type == 'supergat':
            self.convs.append(SuperGATConv(input_dim, hidden_dim, heads=4, dropout=0.6, attention_type='MX', edge_sample_ratio=0.8))
            for _ in range(num_layers - 2):
                self.convs.append(SuperGATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.6, attention_type='MX', edge_sample_ratio=0.8))
            self.convs.append(SuperGATConv(hidden_dim * 4, output_dim, heads=1, concat=False, dropout=0.6, attention_type='MX', edge_sample_ratio=0.8))
        elif gnn_type == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, output_dim))
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
    cut_value = sum(w for u, v, w in edges if solution[u] != solution[v])
    return cut_value

def local_search(solution, edges, max_iter=500):
    n = len(solution)
    best_solution = solution[:]
    best_cut = evaluate_cut(best_solution, edges)
    
    for _ in range(max_iter):
        i, j = random.sample(range(n), 2)
        best_solution[i], best_solution[j] = best_solution[j], best_solution[i]
        new_cut = evaluate_cut(best_solution, edges)
        if new_cut > best_cut:
            best_cut = new_cut
        else:
            best_solution[i], best_solution[j] = best_solution[j], best_solution[i]
    
    return best_solution

def adaptive_perturbation(solution, edges, max_iter=50, alpha=1.0, beta=1.0):
    n = len(solution)
    best_solution = solution[:]
    best_cut = evaluate_cut(best_solution, edges)
    
    for _ in range(max_iter):
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        while i == j:
            j = random.randint(0, n-1)
        
        solution[i], solution[j] = solution[j], solution[i]
        new_cut = evaluate_cut(solution, edges)
        delta_cut = new_cut - best_cut
        
        if delta_cut > 0 or random.random() < math.exp(beta * delta_cut / alpha):
            best_solution = solution[:]
            best_cut = new_cut
        else:
            solution[i], solution[j] = solution[j], solution[i]
        
        alpha *= 0.99
    
    return best_solution

def gnn_guided_search(model, data, solution, edges, max_iter=500):
    n = len(solution)
    best_solution = solution[:]
    best_cut = evaluate_cut(best_solution, edges)
    
    for _ in range(max_iter):
        data.x = torch.tensor(solution, dtype=torch.float).unsqueeze(1).to(device)
        output = model(data).squeeze()
        probabilities = output.detach().cpu().numpy()
        i, j = random.choices(range(n), k=2, weights=probabilities)
        solution[i], solution[j] = solution[j], solution[i]
        new_cut = evaluate_cut(solution, edges)
        if new_cut > best_cut:
            best_cut = new_cut
            best_solution = solution[:]
        else:
            solution[i], solution[j] = solution[j], solution[i]
    
    return best_solution

def simulated_annealing(solution, edges, max_iter=500, initial_temp=1.0, final_temp=0.01, cooling_rate=0.99):
    n = len(solution)
    current_solution = solution[:]
    current_cut = evaluate_cut(current_solution, edges)
    print("Current Cut: " + str(current_cut))
    best_solution = current_solution[:]
    best_cut = current_cut
    
    temp = initial_temp
    while temp > final_temp:
        for _ in range(max_iter):
            i, j = random.sample(range(n), 2)
            current_solution[i], current_solution[j] = current_solution[j], current_solution[i]
            new_cut = evaluate_cut(current_solution, edges)
            delta_cut = new_cut - current_cut
            
            if delta_cut > 0 or random.random() < math.exp(delta_cut / temp):
                current_cut = new_cut
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_solution = current_solution[:]
            else:
                current_solution[i], current_solution[j] = current_solution[j], current_solution[i]
        
        temp *= cooling_rate
    
    return best_solution

def hybrid_approach(edges, n, gnn_model, gnn_data, max_iter=2000, l2a_cut=None):
    population = [np.random.randint(2, size=n) for _ in range(20)]
    best_solution = None
    best_cut = 0
    
    for _ in range(max_iter):
        for i in range(len(population)):
            population[i] = local_search(population[i], edges, max_iter=100)
            population[i] = adaptive_perturbation(population[i], edges, max_iter=25, alpha=1.0, beta=1.0)
            population[i] = gnn_guided_search(gnn_model, gnn_data, population[i], edges, max_iter=50)
            population[i] = simulated_annealing(population[i], edges, max_iter=100, initial_temp=1.0, final_temp=0.01, cooling_rate=0.99)
        
        cuts = [evaluate_cut(solution, edges) for solution in population]
        max_cut = max(cuts)
        if max_cut > best_cut:
            best_cut = max_cut
            best_solution = population[cuts.index(max_cut)]
            
            if l2a_cut is not None and best_cut >= l2a_cut + 1:
                return best_solution, best_cut
        
        # Perform evolutionary operations
        offspring = []
        for _ in range(len(population) // 2):
            parent1, parent2 = random.choices(population, k=2, weights=cuts)
            child1, child2 = uniform_crossover(parent1, parent2, crossover_rate=0.8)
            child1 = bit_flip_mutation(child1, mutation_rate=0.1)
            child2 = bit_flip_mutation(child2, mutation_rate=0.1)
            offspring.append(child1)
            offspring.append(child2)
        
        population = offspring
    
    return best_solution, best_cut

def uniform_crossover(parent1, parent2, crossover_rate=0.8):
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(len(parent1)):
        if random.random() < crossover_rate:
            child1[i] = parent2[i]
            child2[i] = parent1[i]
    return child1, child2

def bit_flip_mutation(solution, mutation_rate=0.1):
    mutated_solution = solution.copy()
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            mutated_solution[i] = 1 - mutated_solution[i]
    return mutated_solution

def train_gnn(edges, n, hidden_dim=32, lr=0.01, epochs=200, gnn_type='supergat'):
    edge_index = torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t().contiguous().to(device)
    x = torch.zeros((n, 1), dtype=torch.float).to(device)
    data = Data(x=x, edge_index=edge_index)
    
    model = AdvancedGNN(1, hidden_dim, 1, num_layers=4, gnn_type=gnn_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    with open('result/my_solution_2/result.txt', 'a') as file:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = nn.BCELoss()(output, x)
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                message = f"Epoch: {epoch}, Loss: {loss.item():.4f}"
                print(message)
                print(message, file=file)
    
    return model

def transfer_learning(model, edges, n, hidden_dim=32, lr=0.01, epochs=100, gnn_type='supergat'):
    edge_index = torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t().contiguous().to(device)
    x = torch.zeros((n, 1), dtype=torch.float).to(device)
    data = Data(x=x, edge_index=edge_index)
    
    # Freeze the weights of the first few layers
    for param in model.convs[:-2].parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    with open('result/qiga/result.txt', 'a') as file:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = nn.BCELoss()(output, x)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                message = f"Transfer Learning Epoch: {epoch}, Loss: {loss.item():.4f}"
                print(message)
                print(message, file=file)
    
    return model

if __name__ == "__main__":
    gset_dir = 'data/gset'
    instances = ['gset_14']
    l2a_results = [3064]
    
    total_l2a_cut = sum(l2a_results)
    total_hybrid_cut = 0
    
    with open('result/qiga/result.txt', 'w') as file:
        pretrained_model = None
        for instance, l2a_cut in zip(instances, l2a_results):
            filename = os.path.join(gset_dir, f'{instance}.txt')
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
            
            if pretrained_model is None:
                gnn_model = train_gnn(edges, n, gnn_type='supergat')
            else:
                gnn_model = transfer_learning(pretrained_model, edges, n, gnn_type='supergat')
            
            pretrained_model = gnn_model
            
            best_cut = 0
            best_solution = None
            
            gnn_data = Data(x=torch.zeros((n, 1), dtype=torch.float).to(device), edge_index=torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t().contiguous().to(device))
            solution, cut = hybrid_approach(edges, n, gnn_model, gnn_data, max_iter=2000, l2a_cut=l2a_cut)
            
            if cut >= l2a_cut + 1:
                best_cut = cut
                best_solution = solution
                
                message = f"Hybrid Approach Cut: {best_cut}"
                print(message)
                print(message, file=file)
                total_hybrid_cut += best_cut
                
                message = f"Improvement %: {100*(best_cut-l2a_cut)/l2a_cut:.2f}"
                print(message)
                print(message, file=file)
                
                message = f"Solution vector: {best_solution}"
                print(message)
                print(message, file=file)
                
                break  # Stop the loop if the condition is met
        
        message = "\nSummary:"
        print(message)
        print(message, file=file)
        
        message = f"Total L2A Cut: {total_l2a_cut}"
        print(message)
        print(message, file=file)
        
        message = f"Total Hybrid Approach Cut: {total_hybrid_cut}"
        print(message)
        print(message, file=file)
        
        message = f"Improvement %: {100*(total_hybrid_cut-total_l2a_cut)/total_l2a_cut:.2f}"
        print(message)
        print(message, file=file)