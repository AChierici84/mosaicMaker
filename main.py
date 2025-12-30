import requests
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

colors = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 255, 0),
    "light gray": (150, 150, 150),
    "dark gray": (86, 86, 86),
}

def download_image(url):
    response = requests.get(url)
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

#funzione mosaico
def mosaic_maker(path:str):
    image=cv2.imread(path)

    # calcolo le dimensioni
    h, w, canali = image.shape

    # determino la più piccola
    size = min(h, w)

    # Calculo le coordinate del centro
    start_x = (w - size) // 2
    start_y = (h - size) // 2

    # taglio l'immagine in un quadrato
    image = image[start_y:start_y + size, start_x:start_x + size]

    """
    CLUSTERING
    """
    data = image.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=len(colors.keys()), random_state=42, n_init='auto')
    kmeans.fit(data)

    labels = kmeans.predict(data)

    # Get the cluster centers (these are in BGR because the input 'image' is BGR)
    kmeans_centers_bgr = kmeans.cluster_centers_

    # Create a list of BGR colors from the predefined 'colors' dictionary
    # The 'colors' dictionary stores RGB, so we convert them to BGR for comparison
    fixed_colors_bgr = []
    for r, g, b in colors.values():
        fixed_colors_bgr.append((b, g, r))
    fixed_colors_bgr = np.array(fixed_colors_bgr, dtype=np.uint8)

    # --- New logic for random unique assignment ---
    num_clusters = len(colors.keys())

    # Create a random permutation of indices for the fixed colors
    random_indices = np.random.permutation(num_clusters)

    new_centers = np.zeros((num_clusters, 3), dtype=np.uint8)

    # Assign each K-Means cluster a unique, randomly chosen fixed color
    for i in range(num_clusters):
        new_centers[i] = fixed_colors_bgr[random_indices[i]]

    # Use these new, fixed centers to create the posterized image
    posterized_image = new_centers[labels]
    posterized_image = posterized_image.reshape(image.shape)

    block_size_final = posterized_image.shape[0] // 48

    # Initialize a true 48x48 mosaic grid
    grid_mosaic_48x48 = np.zeros((48, 48, 3), dtype=np.uint8)
    
                
    # Count the number of pieces needed for each color in the final mosaic
    final_color_counts = {color_name: 0 for color_name in colors.keys()}

    # Create a list of BGR colors from the predefined 'colors' dictionary
    fixed_colors_bgr = []
    for r, g, b in colors.values():
        fixed_colors_bgr.append((b, g, r))
    fixed_colors_bgr = np.array(fixed_colors_bgr, dtype=np.uint8)

    # Iterate through the 48x48 grid
    for i in range(48):
        for j in range(48):
            start_row = i * block_size_final
            end_row = (i + 1) * block_size_final
            start_col = j * block_size_final
            end_col = (j + 1) * block_size_final

            current_block = posterized_image[start_row:end_row, start_col:end_col]
            pixels = current_block.reshape(-1, 3)

            color_votes = {tuple(c.tolist()): 0 for c in fixed_colors_bgr}
            for pixel_color in pixels:
                for fixed_color in fixed_colors_bgr:
                    if np.array_equal(pixel_color, fixed_color):
                        color_votes[tuple(fixed_color.tolist())] += 1
                        break

            dominant_color_bgr = None
            max_count = -1
            for color_bgr_tuple, count in color_votes.items():
                if count > max_count:
                    max_count = count
                    dominant_color_bgr = np.array(color_bgr_tuple, dtype=np.uint8)

            # Store the dominant color directly into the 48x48 grid_mosaic
            if dominant_color_bgr is not None:
                # Convert BGR to RGB for matplotlib display
                grid_mosaic_48x48[i, j] = dominant_color_bgr[[2, 1, 0]] # OpenCV BGR to Matplotlib RGB
                
                # Update final color counts
                # Find the name of the dominant_color_bgr
                for color_name, rgb_value in colors.items():
                    bgr_value = np.array([rgb_value[2], rgb_value[1], rgb_value[0]], dtype=np.uint8) # Convert RGB to BGR array
                    if np.array_equal(dominant_color_bgr, bgr_value):
                        final_color_counts[color_name] += 1
                        break
                

    # Display the 48x48 mosaic grid
    plt.figure(figsize=(10, 10)) # Use a larger figure size for better visibility
    plt.imshow(grid_mosaic_48x48)
    plt.title('Mosaic Grid (48x48 pixels) with Dominant Colors')
    plt.axis('off') # Hide axes
    plt.grid(True)
    plt.show()

    # Print the final color counts
    print("\nNumber of pieces needed for the final mosaic:")
    for color_name, count in final_color_counts.items():
        print(f"- {color_name}: {count}")

    output_image_size = 1440
    num_blocks_per_side = 48
    block_size_px = output_image_size // num_blocks_per_side # 1440 / 48 = 30

    # Initialize the large output image
    final_lego_mosaic = np.zeros((output_image_size, output_image_size, 3), dtype=np.uint8)

    # Get the BGR values for light gray, black, and white
    light_gray_rgb = colors['light gray']
    light_gray_bgr = (light_gray_rgb[2], light_gray_rgb[1], light_gray_rgb[0]) # Convert RGB from dict to BGR for OpenCV

    black_rgb = colors['black']
    black_bgr = (black_rgb[2], black_rgb[1], black_rgb[0])

    white_rgb = colors['white']
    white_bgr = (white_rgb[2], white_rgb[1], white_rgb[0])

    for i in range(num_blocks_per_side): # Iterate through rows of the 48x48 grid
        for j in range(num_blocks_per_side): # Iterate through columns of the 48x48 grid
            # Get the color for the current block from the 48x48 grid (it's in RGB as stored in grid_mosaic_48x48)
            block_color_rgb = grid_mosaic_48x48[i, j]
            # Convert to BGR for OpenCV operations
            block_color_bgr = (block_color_rgb[2], block_color_rgb[1], block_color_rgb[0])

            # Calculate the top-left corner of the block in the large 1440x1440 image
            start_x = j * block_size_px
            start_y = i * block_size_px
            end_x = (j + 1) * block_size_px
            end_y = (i + 1) * block_size_px

            # Fill the entire block area with its dominant color
            final_lego_mosaic[start_y:end_y, start_x:start_x + block_size_px] = block_color_bgr

            # Determine circle color based on block_color_bgr
            circle_color_bgr = light_gray_bgr
            if np.array_equal(block_color_bgr, light_gray_bgr) or np.array_equal(block_color_bgr, black_bgr):
                circle_color_bgr = white_bgr

            # Draw an empty circle in the center of the block
            center_x = start_x + block_size_px // 2
            center_y = start_y + block_size_px // 2
            radius = block_size_px // 2 - 2 # A little smaller than half the block size for padding
            cv2.circle(final_lego_mosaic, (center_x, center_y), radius, circle_color_bgr, 1) # 1 for an empty circle with 1px thickness

    cv2.imwrite(path.split('.')[0]+'_mosaic.png', final_lego_mosaic)
    print("Final Lego mosaic image saved as "+path.split('.')[0]+'_mosaic.png')
    
    # Display the final Lego mosaic
    cv2.imshow('Final Lego Mosaic', final_lego_mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #salva il numero di pezzi in un file di testo
    with open(path.split('.')[0]+'_mosaic_info.txt', 'w') as f:
        f.write("Number of pieces needed for the final mosaic:\n")
        for color_name, count in final_color_counts.items():
            f.write(f"- {color_name}: {count}\n")
            
    print("Mosaic info saved as "+path.split('.')[0]+'_mosaic_info.txt')
    
    return final_lego_mosaic, final_color_counts


# Esempio di utilizzo
if __name__ == "__main__":
    image_url = "https://www.ferrerorocher.com/it/sites/ferrerorocher20_it/files/2021-09/birthday-cake_main.jpg?t=1766141948"  # Sostituisci con l'URL della tua immagine
    image_path = "torta di compleanno.jpg"
    
    # Scarica l'immagine
    image = download_image(image_url)
    cv2.imwrite(image_path, image)
    
    # Crea il mosaico
    mosaic_maker(image_path)
    

