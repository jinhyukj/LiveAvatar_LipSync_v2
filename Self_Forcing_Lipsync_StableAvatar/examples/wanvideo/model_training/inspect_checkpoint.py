import torch

# --- IMPORTANT: Change this path to point to your checkpoint file ---
checkpoint_path = "/home/cvlab20/project/hyunbin/Self-Forcing/checkpoints/self_forcing_dmd.pt"

print(f"--- Inspecting Checkpoint: {checkpoint_path} ---")

try:
    # Load the file
    data = torch.load(checkpoint_path, map_location="cpu")
    
    # Check what type of data was loaded
    print(f"\n1. Type of loaded data: {type(data)}")
    
    # If the data is a dictionary, print its keys
    if isinstance(data, dict):
        print("\n2. It's a dictionary. Here are its keys:")
        print(list(data.keys()))
    else:
        print("\n2. The data is not a dictionary, so it's not a standard state_dict.")

except Exception as e:
    print(f"\nAn error occurred while loading the file: {e}")

print("\n--- Inspection Complete ---")