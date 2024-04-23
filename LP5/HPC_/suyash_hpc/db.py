import multiprocessing
from itertools import permutations
import random
import time
# Simulate the cost evaluation of a query plan
def evaluate_plan(plan):
    print(f"Evaluating plan: {plan}")
    time.sleep(random.uniform(0.5, 1.5)) 
    cost = random.randint(1, 100) 
    return (cost, plan)
# Generate all permutations of joins in a query
def generate_plans(tables):
    return list(permutations(tables))
# Function to be executed in parallel
def optimize_plan(plan):
    return evaluate_plan(plan)
def main():
    tables = ['Table1', 'Table2', 'Table3', 'Table4']
    plans = generate_plans(tables)
    pool = multiprocessing.Pool(processes=4)
    results = pool.map(optimize_plan, plans)
    best_plan = min(results, key=lambda x: x[0])
    print(f"Best plan: {best_plan[1]} with cost: {best_plan[0]}")
    
    # Display costs of all plans
    print("Cost of all plans:")
    for result in results:
        print(f"Plan: {result[1]} with cost: {result[0]}")

if __name__ == "__main__":
    main()
