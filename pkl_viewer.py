import pickle

# Warning: This will use ~2GB+ of RAM
file_path = 'MarsLab Offline RL Feb Transitions.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

if len(data) > 0:
    first_item = data[0]
    
    if isinstance(first_item, dict):
        # 1. Print all available "column" names
        all_keys = list(first_item.keys())
        print(f"Total columns found: {len(all_keys)}")
        print("Available columns:", all_keys)
        print("-" * 30)

        # 2. Define your exclusion list
        exclude = {'obs', 'next_obs'} # Replace with names you saw above

        # 3. Print first two rows with exclusions
        # for i, row in enumerate(data[-1]):
        #     filtered_row = {k: v for k, v in row.items() if k not in exclude}
        #     print(f"\nRow {i} Data:")
        #     print(filtered_row)
        last_row = data[-36]
        filtered_last_row = {k: v for k, v in last_row.items() if k not in exclude}
        print("Last Row Data (Filtered):")
        print(filtered_last_row)
            
    # else:
        # print(f"The file contains a list of {type(first_item)}, not dictionaries.")
        # print("First 2 items:", data[-1])
else:
    print("The pickle file is empty.")