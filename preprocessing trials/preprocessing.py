import cv2
import numpy as np
import os

DISPLAY_FACTOR=0.125

# Load image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'preprocessing_images', 'sample.jpg')
image = cv2.imread(image_path)[1000:2500, 1500:3000]

# Apply CLAHE
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
clahed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# sharpened after clahe
clahed_then_blurred = cv2.GaussianBlur(clahed, (105, 105), 0)
clahed_then_sharpened = cv2.addWeighted(clahed, 2, clahed_then_blurred, -1, 0)


# Apply sharpening
blurred = cv2.GaussianBlur(image, (105, 105), 0)
sharpened = cv2.addWeighted(image, 2, blurred, -1, 0)

# clahe afteer sharpening
lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
sharpened_then_clahed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Resize images for display
resized_image = cv2.resize(image, (0, 0), fx=DISPLAY_FACTOR, fy=DISPLAY_FACTOR)
resized_blurred = cv2.resize(blurred, (0, 0), fx=DISPLAY_FACTOR, fy=DISPLAY_FACTOR)
resized_sharpened = cv2.resize(sharpened, (0, 0), fx=DISPLAY_FACTOR, fy=DISPLAY_FACTOR)

resized_clahed = cv2.resize(clahed, (0, 0), fx=DISPLAY_FACTOR, fy=DISPLAY_FACTOR)
resizrd_clahed_then_blurred = cv2.resize(clahed_then_blurred, (0, 0), fx=DISPLAY_FACTOR, fy=DISPLAY_FACTOR)
resized_clahed_then_sharpened = cv2.resize(clahed_then_sharpened, (0, 0), fx=DISPLAY_FACTOR, fy=DISPLAY_FACTOR)

resized_sharpened_then_clahed = cv2.resize(sharpened_then_clahed, (0, 0), fx=DISPLAY_FACTOR, fy=DISPLAY_FACTOR)


#Display
top_row = np.hstack((resized_image, resized_sharpened))
bottom_row = np.hstack((resized_sharpened_then_clahed, resized_clahed_then_sharpened))
final_layout = np.vstack((top_row, bottom_row))

cv2.imshow("im", final_layout)
cv2.waitKey(0)
cv2.destroyAllWindows()
