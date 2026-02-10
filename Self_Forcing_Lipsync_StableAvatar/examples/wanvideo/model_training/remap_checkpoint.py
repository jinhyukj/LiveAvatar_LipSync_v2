import torch
from safetensors.torch import load_file, save_file
from diffsynth.models.model_manager import ModelManager
import os

# --- Step 1: CONFIGURE YOUR PATHS HERE ---

# Path to your converted checkpoint (the one we made in the last step)
user_checkpoint_path = "converted_checkpoint.safetensors"

# Path for the final, remapped checkpoint that will be used for training
output_folder = os.path.dirname(user_checkpoint_path)
final_checkpoint_path = os.path.join(output_folder, "remapped_checkpoint.safetensors")


# --- Step 2: THE SCRIPT WILL DO THE REST ---

model_manager = ModelManager()

print("Loading user checkpoint...")
user_state_dict = load_file(user_checkpoint_path)
user_keys = list(user_state_dict.keys())
print(f"-> Found {len(user_keys)} parameters.")
print(f"-> First few keys: {user_keys[:5]}")

print("\nLoading original Wan DiT model as a reference...")

# --- CORRECTED LOGIC ---
# 1. Explicitly load the reference model from the hub into the manager.
model_manager.load_model(
    "/home/cvlab20/project/hyunbin/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
)
# 2. Now fetch the loaded model object from the manager.
ref_dit_model = model_manager.fetch_model(
    model_name="wan_video_dit"
)
# ---------------------

ref_state_dict = ref_dit_model.state_dict()
ref_keys = list(ref_state_dict.keys())
print(f"-> Found {len(ref_keys)} parameters.")
print(f"-> First few keys: {ref_keys[:5]}")

# --- Remapping Logic ---

if len(user_keys) != len(ref_keys):
    print(f"\nError: Parameter count mismatch! User model has {len(user_keys)} params, but reference model has {len(ref_keys)}.")
    print("The models are not compatible and cannot be remapped automatically.")
else:
    print("\nParameter counts match. Creating remapped state dictionary...")
    new_state_dict = {}
    for user_key, ref_key in zip(user_keys, ref_keys):
        # Assign the user's weight to the reference model's key name
        new_state_dict[ref_key] = user_state_dict[user_key]
    
    print(f"Saving remapped checkpoint to: {final_checkpoint_path}")
    save_file(new_state_dict, final_checkpoint_path)
    
    print("\n✅ Remapping successful!")