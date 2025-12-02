import matplotlib.pyplot as plt
import re
import argparse
import os

# --- 1. SIMPLE ARGPARSE CONFIGURATION ---
parser = argparse.ArgumentParser(description="Loss visualization script for CycleGAN")

# Add arguments
parser.add_argument('--name', type=str, default='ifr_model_resnet_square', 
                    help='Project name (matches the folder name in checkpoints/)')

# Parse arguments
opt = parser.parse_args()


# --- 2. PATHS AND VARIABLES ---
filename = f'checkpoints/{opt.name}/loss_log.txt'
save_dir = f"results/{opt.name}"

# Create the result directory if it doesn't exist (prevents savefig error)
os.makedirs(save_dir, exist_ok=True)

# Data storage
global_iterations = [] 
losses = {
    'D_A': [], 'G_A': [], 'cycle_A': [], 'idt_A': [],
    'D_B': [], 'G_B': [], 'cycle_B': [], 'idt_B': []
}

# Variables to handle epoch continuity (resetting iterations)
offset = 0
last_iter = 0
epoch_boundaries = [] 

try:
    with open(filename, 'r') as f:
        for line in f:
            # Skip lines that don't contain iteration info
            if "iters:" not in line:
                continue

            # Extract current iteration number
            iter_match = re.search(r'iters:\s+(\d+)', line)
            if iter_match:
                current_iter = int(iter_match.group(1))
                
                # --- EPOCH DETECTION LOGIC ---
                # If the current iteration is smaller than the previous one,
                # it means a new epoch has started. We update the offset.
                if current_iter < last_iter:
                    offset += last_iter
                    epoch_boundaries.append(offset)
                
                # Calculate global iteration (absolute/cumulative)
                global_iter = current_iter + offset
                global_iterations.append(global_iter)
                
                # Update last known iteration
                last_iter = current_iter
            
                # Extract loss values
                for key in losses.keys():
                    loss_match = re.search(rf'{key}:\s+([\d\.]+)', line)
                    if loss_match:
                        losses[key].append(float(loss_match.group(1)))
                    else:
                        losses[key].append(None)

    # --- CONVERT ITERATIONS TO EPOCHS ---
    # The number of iterations in the first epoch determines the scale
    iters_per_epoch = epoch_boundaries[0]+80 if epoch_boundaries else (global_iterations[-1] if global_iterations else 1)
    # Convert the list of global iterations to a list of corresponding epoch numbers
    epochs = [i / iters_per_epoch for i in global_iterations]
    total_epochs = epochs[-1] if epochs else 0
    print(total_epochs)
    # --- PLOTTING ---
    # Create 4 stacked subplots sharing the same X-axis
    fig, (ax_id, ax_gen, ax_cycle, ax_disc) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # Define groups for each subplot
    group_id    = ['idt_A', 'idt_B']
    group_gen   = ['G_A', 'G_B']         # Pure adversarial loss
    group_cycle = ['cycle_A', 'cycle_B'] # Cycle consistency loss
    group_disc  = ['D_A', 'D_B']

    # Helper function to plot on a specific axis
    def plot_on_axis(ax, keys, max_val, title):
        for key in keys:
            values = losses[key]
            # Filter out None values to keep lines clean
            valid_data = [(e, v) for e, v in zip(epochs, values) if v is not None]
            if valid_data:
                valid_epochs, valid_vals = zip(*valid_data)
                ax.plot(valid_epochs, valid_vals, label=key, linewidth=1.5, alpha=0.8)
        
        ax.set_ylim(0, max_val)
        ax.set_xlim(0, total_epochs)
        
        # Draw vertical lines every 50 epochs
        for epoch_mark in range(0, int(total_epochs) + 1, 50):
            ax.axvline(x=epoch_mark, color='black', linestyle='--', alpha=0.4)
            
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel("Loss")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

    # Plotting the 4 distinct levels with custom y-axis scales
    plot_on_axis(ax_id,    group_id,    0.05,   "1. Identity Losses (Color Preservation)")
    plot_on_axis(ax_cycle, group_cycle, 0.15,  "2. Cycle Consistency Losses (Reconstruction)")
    plot_on_axis(ax_gen,   group_gen,   0.4,   "3. Generator Adversarial Losses (Fooling D)")
    plot_on_axis(ax_disc,  group_disc,  0.35,   "4. Discriminator Losses (Real vs Fake)")

    # Set X-label only on the bottom graph
    ax_disc.set_xlabel(f"Epochs (Total: {int(total_epochs)})", fontsize=12)

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(save_dir, "cyclegan_losses_4_levels.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
