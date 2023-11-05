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
# result_np = np.random.randint(min_value, max_value + 1, size=(num_rows, num_cols, num_elements))
# result_list = result_np.tolist()
# with open("serialized_array.pkl", "wb") as file:
#     pickle.dump(result_np, file)

# Serialize the list as JSON and save it to a file
# with open("serialized_array.json", "w") as file:
#     json.dump(result_list, file)

def truncate_and_split_lines(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if len(line) <= 1000:
                outfile.write(line)
            else:
                while len(line) > 1000:
                    split_point = line.rfind('\\', 0, 1000)
                    if split_point == -1:
                        # If no suitable split point with '\' found, split at exactly 3000 characters.
                        split_point = 1000
                    outfile.write(line[:split_point] + '\\\n')
                    line = line[split_point:]
                outfile.write(line)

# Example usage:
input_file = 'serialized_array.json'
output_file = 'pretty_serialized_array.json'
truncate_and_split_lines(input_file, output_file)


# # Serialize the list to a binary string
# serialized_data = pickle.dumps(result_np)
#
# # memory_usage = sys.getsizeof(result)
# print(serialized_data)
# print(f"Memory usage for one element: {result_np.nbytes} bytes")

# memory_usage = sys.getsizeof(result)
# print(f" Second - Memory usage for one element: {memory_usage} bytes")

