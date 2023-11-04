import json
import sys
import pickle
import numpy as np

num_rows = 7000
num_cols = 3000
num_elements = 8
min_value = 0
max_value = 10  # Change this range as needed

# Create a list of 7000*3000 elements, each element being a list of 8 elements
# result = [[0] * 8 for _ in range(7000*3000)]
result_np = np.random.randint(min_value, max_value + 1, size=(num_rows, num_cols, num_elements))
result_list = result_np.tolist()
# with open("serialized_array.pkl", "wb") as file:
#     pickle.dump(result_np, file)

# Serialize the list as JSON and save it to a file
with open("serialized_array.json", "w") as file:
    json.dump(result_list, file)


# # Serialize the list to a binary string
# serialized_data = pickle.dumps(result_np)
#
# # memory_usage = sys.getsizeof(result)
# print(serialized_data)
# print(f"Memory usage for one element: {result_np.nbytes} bytes")

# memory_usage = sys.getsizeof(result)
# print(f" Second - Memory usage for one element: {memory_usage} bytes")

