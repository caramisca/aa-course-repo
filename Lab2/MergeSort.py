import random
import time
import tracemalloc
import pandas as pd
import matplotlib.pyplot as plt

def merge_sort(array):
    if len(array) <= 1:
        return array

    middle = len(array) // 2
    left_array = array[:middle]
    right_array = array[middle:]

    return merge(merge_sort(left_array), merge_sort(right_array))

def merge(left, right):
    sorted_array = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_array.append(left[i])
            i += 1
        else:
            sorted_array.append(right[j])
            j += 1
    return sorted_array + left[i:] + right[j:]

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
        merge_sort(array)
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
plt.title('Merge Sort Execution Time for Different Datasets')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()
