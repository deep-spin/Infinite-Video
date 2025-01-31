import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import shutil
import numpy as np
from PIL import Image

with open('./alphas_uniform', 'rb') as f:
    density_tensor = pickle.load(f)

# Assuming density_tensor_uniform and density_tensor_sticky are loaded and contain the required data
# Convert the loaded lists to NumPy arrays if necessary
density_tensor_sticky = np.array(density_tensor.cpu())

density_sticky = np.mean(density_tensor_sticky, axis=(0, 1, 2))

density_sticky = density_sticky/np.sum(density_sticky)
# Define chunk sizes
chunk_size = 256
chunks = [range(i, i + chunk_size) for i in range(0, 768, chunk_size)]  # Chunk indices

# Plotting: create 2 rows of 3 plots for uniform and sticky
fig, axs = plt.subplots(1, 3, figsize=(12, 1.5), constrained_layout=True)

# Loop over each chunk and plot both uniform and sticky
for i, chunk in enumerate(chunks):
    # Extract corresponding ranges from the uniform and sticky density arrays
    sticky_chunk = density_sticky[chunk]
    # Top row: uniform density
    sns.heatmap(sticky_chunk.reshape(1, -1), cmap="viridis", cbar=True, 
                ax=axs[i], square=False, yticklabels=False, cbar_kws={'orientation': 'vertical'})

    # Set xticks to match the chunk range
    xtick_positions = np.linspace(0, 256, 6)  # Adjust this depending on the number of ticks you want
    xtick_labels = np.round(np.linspace(chunk.start, chunk.stop, 6), 0).astype(int)  # Labels for the chunk range
    axs[i].set_xticks(xtick_positions)
    axs[i].set_xticklabels(xtick_labels, fontsize=10, rotation=0)
    axs[i].set_xlabel("# Frames", fontsize=10)


# Save and display the figure
output_path = "chunks.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

import torch
video_list = torch.load("your_video")
k = 10
frames_dir = "frames_uniform"
for i, chunk in enumerate(chunks):
    # Extract corresponding ranges from the uniform and sticky density arrays
    sticky_chunk = density_sticky[chunk]
    
    # Get the top-k indices for the sticky density (descending order)
    top_k_sticky_indices = np.argsort(sticky_chunk)[-k:][::-1]
    
    # Print the indices for each chunk
    print(f"Chunk {i + 1}: {chunk.start} to {chunk.stop - 1}")
    print(f"Top {k} sticky density indices: {top_k_sticky_indices}")
    print("-" * 50)
    
    # Process the top-k uniform density frames
    for idx in top_k_sticky_indices:
        try:
            # Retrieve the video tensor for the corresponding index (assuming video_list is available)
            video_tensor = video_list[:, idx + 256*i]  # Retrieve the image path from the video list
            
            # Convert the tensor to a numpy array (HWC format)
            image_np = video_tensor.permute(1,2, 0).cpu().numpy()  # Change shape to HWC (Height, Width, Channels)
            
            # Normalize values if necessary (assuming the tensor values are between 0 and 1)
            image = Image.fromarray(image_np.astype(np.uint8))
            
            
            # Construct the filename for saving
            filename = os.path.join(frames_dir, f"frame_{i + 1}_{idx + 256*i}.png")
            
            # Save the image as PNG
            image.save(filename)
        except Exception as e:
            print(f"Failed to load or save uniform image for index {idx}: {e}")