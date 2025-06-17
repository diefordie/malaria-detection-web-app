import cv2
import numpy as np
import pandas as pd
from skimage.measure import label as sklabel, regionprops
from skimage.feature import graycomatrix, graycoprops

def extract_features_with_watershed(json_data, img_folder):
    results = []

    for sample in json_data:
        image_path = img_folder + sample['image']['pathname']
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # --- Watershed Segmentation ---
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
        _, markers = cv2.connectedComponents(np.uint8(sure_fg))
        markers = markers + 1
        markers[unknown == 255] = 0
        markers_colored = img.copy()
        markers_watershed = cv2.watershed(markers_colored, markers)

        # Hasil boundary (optional buat ditampilkan)
        boundary = np.zeros_like(opening)
        boundary[markers_watershed == -1] = 255

        # --- Fitur extraction ---
        area_list, perimeter_list, eccentricity_list, roundness_list = [], [], [], []
        contrast_list, homogeneity_list, correlation_list = [], [], []
        intensity_list = []
        parasite_found = 0

        for obj in sample['objects']:
            category = obj['category']
            x1, y1 = obj['bounding_box']['minimum']['c'], obj['bounding_box']['minimum']['r']
            x2, y2 = obj['bounding_box']['maximum']['c'], obj['bounding_box']['maximum']['r']

            crop_binary = opening[y1:y2, x1:x2]
            crop_gray = blurred[y1:y2, x1:x2]

            if crop_binary.size == 0 or np.count_nonzero(crop_binary) == 0:
                continue

            labeled = sklabel(crop_binary)
            props = regionprops(labeled)

            for prop in props:
                area = prop.area
                perimeter = prop.perimeter
                ecc = prop.eccentricity
                roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

                area_list.append(area)
                perimeter_list.append(perimeter)
                eccentricity_list.append(ecc)
                roundness_list.append(roundness)

            # Texture (GLCM)
            if crop_gray.size > 0:
                glcm = graycomatrix(crop_gray, [1], [0], symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')[0,0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
                correlation = graycoprops(glcm, 'correlation')[0,0]

                contrast_list.append(contrast)
                homogeneity_list.append(homogeneity)
                correlation_list.append(correlation)

            mean_intensity = np.mean(crop_gray)
            intensity_list.append(mean_intensity)

            if category not in ['red blood cell', 'leukocyte']:
                parasite_found += 1

        def safe_mean(lst):
            return np.mean(lst) if len(lst) > 0 else 0

        results.append({
            'image_path': sample['image']['pathname'],
            'mean_area': safe_mean(area_list),
            'mean_perimeter': safe_mean(perimeter_list),
            'mean_eccentricity': safe_mean(eccentricity_list),
            'mean_roundness': safe_mean(roundness_list),
            'mean_contrast': safe_mean(contrast_list),
            'mean_homogeneity': safe_mean(homogeneity_list),
            'mean_correlation': safe_mean(correlation_list),
            'mean_intensity': safe_mean(intensity_list),
            'label': 1 if parasite_found > 0 else 0
        })

    return pd.DataFrame(results)
