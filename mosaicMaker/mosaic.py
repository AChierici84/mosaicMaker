import requests
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


default_colors = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 255, 0),
    "light gray": (150, 150, 150),
    "dark gray": (86, 86, 86),
}

lego_colors = {
    "Bright_Red":(180, 0, 0),
    "Reddish_Orange":(202, 76, 11),
    "Nougat":(187, 128, 90),
    "Dark_Orange":(145, 80, 28),
    "Light_Nougat":(225, 190, 161),
    "Reddish_Brown":(95, 49, 9),
    "Medium_Nougat":(170, 125, 85),
    "Bright_Orange":(214, 121, 35),
    "Dark_Brown":(55, 33, 0),
    "Flame_Yellowish_Orange":(252, 172, 0),
    "Sand_Yellow":(137, 125, 98),
    "Brick_Yellow":(204, 185, 141),
    "Warm_Gold":(185, 149, 59),
    "Bright_Yellow":(250, 200, 10),
    "Cool_Yellow":(255, 236, 108),
    "Olive_Green":(119, 119, 78),
    "Vibrant_Yellow":(255, 255, 0),
    "Bright_Yellowish_Green":(165, 202, 24),
    "Spring_Yellowish_Green":(226, 249, 154),
    "Bright_Green":(88, 171, 65),
    "Dark_Green":(0, 133, 43),
    "Earth_Green":(0, 69, 26),
    "Sand_Green":(112, 142, 124),
    "Aqua":(211, 242, 234),
    "Bright_Bluish_Green":(0, 152, 148),
    "Medium_Azur":(104, 195, 226),
    "Dark_Azur":(70, 155, 195),
    "Black":(27, 42, 52),
    "Bright_Blue":(30, 90, 168),
    "Light_Royal_Blue":(157, 195, 247),
    "Medium_Blue":(115, 150, 200),
    "Sand_Blue":(112, 129, 154),
    "Earth_Blue":(25, 50, 90),
    "Medium_Lilac":(68, 26, 145),
    "Medium_Lavender":(160, 110, 185),
    "Lavender":(205, 164, 222),
    "Bright_Reddish_Violet":(144, 31, 118),
    "Bright_Purple":(200, 80, 155),
    "Light_Purple":(255, 158, 205),
    "New_Dark_Red":(114, 0, 18),
    "Vibrant_Coral":(240, 109, 120),
    "White":(244, 244, 244),
    "Medium_Stone_Grey":(150, 150, 150),
    "Dark_Stone_Grey":(100, 100, 100)
}

#funzione mosaico
def mosaic_maker(path:str, colors:dict=default_colors, output_size:int=1440, num_blocks_per_side:int=48):
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

    # recupero i centri dei cluster
    kmeans_centers_bgr = kmeans.cluster_centers_

    # creo una lista di colori fissi in BGR
    fixed_colors_bgr = []
    for r, g, b in colors.values():
        fixed_colors_bgr.append((b, g, r))
    fixed_colors_bgr = np.array(fixed_colors_bgr, dtype=np.uint8)

    # Numero di cluster
    num_clusters = len(colors.keys())

    # Genera una permutazione casuale degli indici dei cluster
    random_indices = np.random.permutation(num_clusters)

    new_centers = np.zeros((num_clusters, 3), dtype=np.uint8)

    # Assegna i colori fissi ai nuovi centri in base alla permutazione casuale
    for i in range(num_clusters):
        new_centers[i] = fixed_colors_bgr[random_indices[i]]

    # Crea l'immagine posterizzata sostituendo i colori originali con i nuovi centri
    posterized_image = new_centers[labels]
    posterized_image = posterized_image.reshape(image.shape)

    block_size_final = posterized_image.shape[0] // num_blocks_per_side

    # inizializzo la griglia 
    grid_mosaic = np.zeros((num_blocks_per_side, num_blocks_per_side, 3), dtype=np.uint8)
    
                
    # Initialize final color counts
    final_color_counts = {color_name: 0 for color_name in colors.keys()}

    # creo una lista di colori fissi in BGR
    fixed_colors_bgr = []
    for r, g, b in colors.values():
        fixed_colors_bgr.append((b, g, r))
    fixed_colors_bgr = np.array(fixed_colors_bgr, dtype=np.uint8)

    # itero su ogni blocco 
    for i in range(num_blocks_per_side):
        for j in range(num_blocks_per_side):
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

            # salvo il colore dominante nella griglia
            if dominant_color_bgr is not None:
                # Converti BGR a RGB per Matplotlib
                grid_mosaic[i, j] = dominant_color_bgr[[2, 1, 0]] # OpenCV BGR to Matplotlib RGB
                
                # aggiorno il conteggio dei colori finali
                # Trova il nome del colore corrispondente
                for color_name, rgb_value in colors.items():
                    bgr_value = np.array([rgb_value[2], rgb_value[1], rgb_value[0]], dtype=np.uint8) # Convert RGB to BGR array
                    if np.array_equal(dominant_color_bgr, bgr_value):
                        final_color_counts[color_name] += 1
                        break
                

    # mostra la griglia 
    plt.figure(figsize=(10, 10)) # Set figure size
    plt.imshow(grid_mosaic)
    plt.title('Mosaic Grid (' + str(num_blocks_per_side) + 'x' + str(num_blocks_per_side) + ' pixels) with Dominant Colors')
    plt.axis('off') # nasocndo gli assi
    plt.show()

    #  stampo il numero di pezzi necessari per ogni colore
    print("\nNumber of pieces needed for the final mosaic:")
    for color_name, count in final_color_counts.items():
        print(f"- {color_name}: {count}")
        
    """
    CREAZIONE IMMAGINE FINALE output_sizexoutput_size CON CERCHI
    """

    output_image_size = output_size
    block_size_px = output_image_size // num_blocks_per_side 

    # Inizializzo l'immagine finale
    final_lego_mosaic = np.zeros((output_image_size, output_image_size, 3), dtype=np.uint8)

    # Prepara i colori necessari in BGR
    light_gray_rgb = colors['light gray']
    light_gray_bgr = (light_gray_rgb[2], light_gray_rgb[1], light_gray_rgb[0]) # Convert to BGR

    black_rgb = colors['black']
    black_bgr = (black_rgb[2], black_rgb[1], black_rgb[0])

    white_rgb = colors['white']
    white_bgr = (white_rgb[2], white_rgb[1], white_rgb[0])

    for i in range(num_blocks_per_side): # itero sui righe della griglia
        for j in range(num_blocks_per_side): # itero sulle colonne della griglia 
            # Recupero il colore dominante del blocco
            block_color_rgb = grid_mosaic[i, j]
            # Converti RGB a BGR
            block_color_bgr = (block_color_rgb[2], block_color_rgb[1], block_color_rgb[0])

            # calcolo le coordinate del blocco nell'immagine finale
            start_x = j * block_size_px
            start_y = i * block_size_px
            end_x = (j + 1) * block_size_px
            end_y = (i + 1) * block_size_px

            # Riempio il blocco con il colore dominante
            final_lego_mosaic[start_y:end_y, start_x:start_x + block_size_px] = block_color_bgr

            # Determino il colore del cerchio
            circle_color_bgr = light_gray_bgr
            if np.array_equal(block_color_bgr, light_gray_bgr) or np.array_equal(block_color_bgr, black_bgr):
                circle_color_bgr = white_bgr

            # Disegno il cerchio al centro del blocco
            center_x = start_x + block_size_px // 2
            center_y = start_y + block_size_px // 2
            radius = block_size_px // 2 - 2 # un po' di margine
            cv2.circle(final_lego_mosaic, (center_x, center_y), radius, circle_color_bgr, 1) # cerchio vuoto con 1 pixel di spessore 

    cv2.imwrite(path.split('.')[0]+'_mosaic.png', final_lego_mosaic)
    print("Final Lego mosaic image saved as "+path.split('.')[0]+'_mosaic.png')
    
    # mostra l'immagine finale
    cv2.imshow('Final Lego Mosaic', final_lego_mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # salvo le informazioni sui pezzi in un file di testo
    with open(path.split('.')[0]+'_mosaic_info.txt', 'w') as f:
        f.write("Number of pieces needed for the final mosaic:\n")
        for color_name, count in final_color_counts.items():
            f.write(f"- {color_name}: {count}\n")
            
    print("Mosaic info saved as "+path.split('.')[0]+'_mosaic_info.txt')
    
    return final_lego_mosaic, final_color_counts

#funzione mosaico a colori
def mosaic_maker_color(path:str, num_colors:5, output_size:int=1440, num_blocks_per_side:int=48):
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

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init='auto')
    kmeans.fit(data)

    labels = kmeans.predict(data)


    # Numero di cluster
    num_clusters = num_colors#len(colors.keys())


    # sostituisco i colori originali con i centri dei cluster a meno dei decimali
    posterized_image = np.round(kmeans.cluster_centers_[labels]).astype(np.uint8)
    posterized_image = posterized_image.reshape(image.shape)

    # mostra l'immagine posterizzata
    posterized_image_rgb = cv2.cvtColor(posterized_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(posterized_image_rgb)
    plt.title('Posterized Image')
    plt.axis('off')
    plt.show()
    
    block_size_final = posterized_image.shape[0] // num_blocks_per_side

    # inizializzo la griglia
    grid_mosaic = np.zeros((num_blocks_per_side, num_blocks_per_side, 3), dtype=np.uint8)
    
    # prendo i colori dei lego più vicini ai centri dei cluster (dopo averli trasformati in RGB)
    centers_rgb = kmeans.cluster_centers_[:, [2, 1, 0]]  # Converti BGR a RGB
    
    fixed_colors_bgr = []
    used_colors = {}
    for i, center_rgb in enumerate(centers_rgb):
        distances = [np.linalg.norm(center_rgb - np.array(lego_rgb)) for lego_rgb in lego_colors.values()]
        min_idx = np.argmin(distances)
        closest_color_name = list(lego_colors.keys())[min_idx]
        closest_rgb = list(lego_colors.values())[min_idx]
        closest_bgr = (closest_rgb[2], closest_rgb[1], closest_rgb[0])
        fixed_colors_bgr.append(closest_bgr)
        used_colors[closest_color_name] = closest_rgb
    
    fixed_colors_bgr = np.array(fixed_colors_bgr, dtype=np.uint8)
    colors = used_colors
    
    # sostituisco i colori originali con i colori lego più vicini
    posterized_image = fixed_colors_bgr[labels]
    posterized_image = posterized_image.reshape(image.shape)

    # Initialize final color counts
    final_color_counts = {name: 0 for name in colors.keys()}

    # itero su ogni blocco 
    for i in range(num_blocks_per_side):
        for j in range(num_blocks_per_side):
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

            # salvo il colore dominante nella griglia
            if dominant_color_bgr is not None:
                # Converti BGR a RGB per Matplotlib
                grid_mosaic[i, j] = dominant_color_bgr[[2, 1, 0]] # OpenCV BGR to Matplotlib RGB
                
                # aggiorno il conteggio dei colori finali
                # Trova il nome del colore corrispondente
                for color_name, rgb_value in colors.items():
                    bgr_value = np.array([rgb_value[2], rgb_value[1], rgb_value[0]], dtype=np.uint8) # Convert RGB to BGR array
                    if np.array_equal(dominant_color_bgr, bgr_value):
                        final_color_counts[color_name] += 1
                        break
                

    # mostra la griglia
    plt.figure(figsize=(10, 10)) # Set figure size
    plt.imshow(grid_mosaic)
    plt.title('Mosaic Grid (' + str(num_blocks_per_side) + 'x' + str(num_blocks_per_side) + ' pixels) with Dominant Colors')
    plt.axis('off') # nasocndo gli assi
    plt.show()
    
    #  stampo i codici dei colori usati
    print("Colors used in the mosaic:")
    for color_name, rgb_value in colors.items():
        print(f"- {color_name}: RGB{rgb_value}")

    #  stampo il numero di pezzi necessari per ogni colore
    print("\nNumber of pieces needed for the final mosaic:")
    for color_name, count in final_color_counts.items():
        print(f"- {color_name}: {count}")
        
    """
    CREAZIONE IMMAGINE FINALE output_sizexoutput_size CON CERCHI
    """

    output_image_size = output_size
    block_size_px = output_image_size // num_blocks_per_side 

    # Inizializzo l'immagine finale
    final_lego_mosaic = np.zeros((output_image_size, output_image_size, 3), dtype=np.uint8)

    # Prepara i colori necessari in BGR
    light_gray_rgb = default_colors['light gray']
    light_gray_bgr = (light_gray_rgb[2], light_gray_rgb[1], light_gray_rgb[0]) # Convert to BGR

    black_rgb = default_colors['black']
    black_bgr = (black_rgb[2], black_rgb[1], black_rgb[0])

    white_rgb = default_colors['white']
    white_bgr = (white_rgb[2], white_rgb[1], white_rgb[0])

    for i in range(num_blocks_per_side): # itero sui righe della griglia
        for j in range(num_blocks_per_side): # itero sulle colonne della griglia
            # Recupero il colore dominante del blocco
            block_color_rgb = grid_mosaic[i, j]
            # Converti RGB a BGR
            block_color_bgr = (block_color_rgb[2], block_color_rgb[1], block_color_rgb[0])

            # calcolo le coordinate del blocco nell'immagine finale
            start_x = j * block_size_px
            start_y = i * block_size_px
            end_x = (j + 1) * block_size_px
            end_y = (i + 1) * block_size_px

            # Riempio il blocco con il colore dominante
            final_lego_mosaic[start_y:end_y, start_x:start_x + block_size_px] = block_color_bgr

            # Determino il colore del cerchio
            circle_color_bgr = light_gray_bgr
            if np.array_equal(block_color_bgr, light_gray_bgr) or np.array_equal(block_color_bgr, black_bgr):
                circle_color_bgr = white_bgr

            # Disegno il cerchio al centro del blocco
            center_x = start_x + block_size_px // 2
            center_y = start_y + block_size_px // 2
            radius = block_size_px // 2 - 2 # un po' di margine
            cv2.circle(final_lego_mosaic, (center_x, center_y), radius, circle_color_bgr, 1) # cerchio vuoto con 1 pixel di spessore 

    cv2.imwrite(path.split('.')[0]+'_mosaic.png', final_lego_mosaic)
    print("Final Lego mosaic image saved as "+path.split('.')[0]+'_mosaic.png')
    
    # mostra l'immagine finale
    cv2.imshow('Final Lego Mosaic', final_lego_mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # salvo le informazioni sui pezzi in un file di testo
    with open(path.split('.')[0]+'_mosaic_info.txt', 'w') as f:
        #  stampo i codici dei colori usati
        f.write("Colors used in the mosaic:\n")
        for color_name, rgb_value in colors.items():
            f.write(f"- {color_name}: RGB{rgb_value}\n")
        
        f.write("Number of pieces needed for the final mosaic:\n")
        for color_name, count in final_color_counts.items():
            f.write(f"- {color_name}: {count}\n")
        
            
    print("Mosaic info saved as "+path.split('.')[0]+'_mosaic_info.txt')
    
    return final_lego_mosaic, final_color_counts

