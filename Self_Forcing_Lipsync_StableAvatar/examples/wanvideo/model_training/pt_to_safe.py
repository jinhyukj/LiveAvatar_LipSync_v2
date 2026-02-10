import torch
from safetensors.torch import save_file
from collections import OrderedDict
from diffsynth.models.model_manager import ModelManager

# --- Step 1: CONFIGURE YOUR PATHS HERE ---
original_pt_path = "/home/cvlab20/project/hyunbin/Self-Forcing/checkpoints/self_forcing_dmd.pt"
reference_wan_dit_path = "/home/cvlab20/project/hyunbin/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
final_checkpoint_path = "wan_dit_from_self_forcing_CORRECTED.safetensors"

# --- Step 2: (Optional) Manual mapping for any exceptions ---
# We will leave this empty for now and let the script try to map everything automatically.
MANUAL_KEY_MAPPING = {
    # "new_wan_key": "old_self_forcing_key_if_different"
}


# --- Step 3: THE SCRIPT WILL DO THE REST ---
print("Loading original .pt checkpoint...")
original_data = torch.load(original_pt_path, map_location="cpu")
original_state_dict = original_data['generator_ema']

print("Loading reference Wan DiT model for its structure...")
model_manager = ModelManager()
model_manager.load_model(reference_wan_dit_path)
ref_dit_model = model_manager.fetch_model(model_name="wan_video_dit")
ref_state_dict = ref_dit_model.state_dict()

new_state_dict = OrderedDict()
unmapped_keys = []
mismatched_shapes = []

print("\nStarting automated remapping based on 'model.' prefix pattern...")
for ref_key in ref_state_dict.keys():
    old_key_guess = "model." + ref_key
    
    # First, check for a manual override
    if ref_key in MANUAL_KEY_MAPPING:
        old_key_guess = MANUAL_KEY_MAPPING[ref_key]
        
    # Now, check if our guessed key exists and has the correct shape
    if old_key_guess in original_state_dict:
        if original_state_dict[old_key_guess].shape == ref_state_dict[ref_key].shape:
            new_state_dict[ref_key] = original_state_dict[old_key_guess]
        else:
            mismatched_shapes.append(
                f"  - Mismatch for '{ref_key}': "
                f"Expected {ref_state_dict[ref_key].shape}, "
                f"but found {original_state_dict[old_key_guess].shape} in old key '{old_key_guess}'"
            )
    else:
        unmapped_keys.append(ref_key)

# --- Final Report ---
print("\n--- Remapping Report ---")
if not unmapped_keys and not mismatched_shapes:
    print(f"✅ Success! All {len(ref_state_dict)} keys were automatically mapped.")
    print(f"Saving final checkpoint to: {final_checkpoint_path}")
    save_file(new_state_dict, final_checkpoint_path)
    print("\nYou can now use this new file with your training script.")
else:
    print("❌ Found some issues. Please review the details below.")
    if unmapped_keys:
        print(f"\nCould not find a match for {len(unmapped_keys)} keys:")
        for key in unmapped_keys[:10]:
            print(f"  - {key}")
        print("  (and potentially more...)")
        print("  -> You may need to add these to the MANUAL_KEY_MAPPING.")
    if mismatched_shapes:
        print(f"\nFound {len(mismatched_shapes)} keys with mismatched shapes:")
        for error in mismatched_shapes[:10]:
            print(error)