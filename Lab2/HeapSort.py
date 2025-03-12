import random
import time
import tracemalloc
import pandas as pd
import matplotlib.pyplot as plt

# Heap sort functions
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements from the heap one by one
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # Swap
        heapify(arr, i, 0)

# Generate Different Datasets
n = 100000  # Large dataset size

datasets = [
    ("Random Large Dataset", [random.randint(1, 1000000) for _ in range(n)]),
    ("Nearly Sorted Dataset", sorted([random.randint(1, 1000000) for _ in range(n)])),
    ("Small Dataset", [random.randint(1, 100) for _ in range(50)]),
    ("Integer Limited Range (1-100)", [random.randint(1, 100) for _ in range(n)]),
    ("Floating-Point Dataset", [random.uniform(0.0, 1000000.0) for _ in range(n)])  # Works directly with floats
]

sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

# Data to be collected for the table and plot
all_data = []
plt.figure(figsize=(10, 6))

for label, dataset in datasets:
    execution_times = []
    memory_usages = []

    for size in sizes:
        array = dataset[:size]  # Use the corresponding slice of each dataset

        tracemalloc.start()  # Start tracking memory
        start_time = time.time()
        heap_sort(array)  # Use heap sort
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()  # Get memory usage
        tracemalloc.stop()  # Stop tracking memory

        execution_time = end_time - start_time
        peak_memory_mb = peak / (1024 ** 2)  # Convert to MB

        execution_times.append(execution_time)
        memory_usages.append(peak_memory_mb)

        all_data.append([label, size, execution_time, peak_memory_mb])

    # Plot execution time
    plt.plot(sizes, execution_times, marker='o', linestyle='-', label=f"{label} (Time)")

# Create DataFrame, filter to only include results for size 100000
filtered_data = [row for row in all_data if row[1] == 100000]
df = pd.DataFrame(filtered_data, columns=["Dataset Type", "Array Size", "Execution Time (seconds)", "Peak Memory (MB)"])

# Set pandas options to display the entire DataFrame
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # No line width limit
pd.set_option('display.max_colwidth', None)  # No column width limit

# Display the table
print(df)

# Plot customization
plt.xlabel('Array Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Heap Sort Execution Time for Different Datasets')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()
