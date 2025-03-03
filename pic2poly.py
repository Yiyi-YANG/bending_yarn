import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import skimage.morphology as morphology
import skimage.measure as measure
from scipy.interpolate import interp1d

# Set high DPI for clear display
plt.rcParams['figure.dpi'] = 100

# ---------------------------
# Part 1: Image Loading and Preprocessing
# ---------------------------

# Load the original image (skimage reads in RGB)
image_path = r"E:\bending\DCIM\100NZ6_3\DSC_0223.JPG"
image = io.imread(image_path)

# Define region of interest (ROI) in original image
# (rows: 2000 to 2500, columns: 2500 to 5000)
x0, x1, y0, y1 = 2800, 3500, 2000, 2300
roi = image[y0:y1, x0:x1]

# Display the cropped ROI for context
plt.figure(figsize=(roi.shape[1]/100, roi.shape[0]/100))
plt.imshow(roi, interpolation='nearest')
plt.title("Cropped ROI")
plt.show()

# Convert ROI to HSV and create a black mask to detect dark regions (the yarn)
hsv_roi = color.rgb2hsv(roi)
value_thresh = 0.2    # threshold for low brightness
saturation_thresh = 0.3  # threshold for low saturation
black_mask = (hsv_roi[:, :, 2] < value_thresh) & (hsv_roi[:, :, 1] < saturation_thresh)

# Apply morphological closing to clean up the mask
black_mask_cleaned = morphology.closing(black_mask, morphology.square(5))

# Display the processed black mask
plt.figure(figsize=(roi.shape[1]/100, roi.shape[0]/100))
plt.imshow(black_mask_cleaned, cmap='gray')
plt.title("Processed Black Mask")
plt.show()

# ---------------------------
# Part 2: Manual Start Point and Ordered Contour Extraction
# ---------------------------

# Let user manually click the yarn START point on the ROI.
# Note: ginput returns [x, y] coordinates.
plt.figure(figsize=(roi.shape[1]/100, roi.shape[0]/100))
plt.imshow(roi, interpolation='nearest')
plt.title("Click the Yarn START Point; Press ENTER after 1 click")
plt.show(block=False)
manual_click = np.array(plt.ginput(1, timeout=0))[0]  # [x, y] in ROI coordinates
plt.close()
print(f"Manual Start Point (ROI, [x,y]): {manual_click}")

# Extract entire yarn contour from the black mask using find_contours.
# find_contours returns an ordered list of (row, col) coordinates.
contours = measure.find_contours(black_mask_cleaned.astype(float), level=0.5)
if len(contours) == 0:
    print("No yarn contour found. Please adjust thresholds.")
    exit()

# Select the longest contour as the yarn contour.
selected_contour = max(contours, key=len)

# Convert manual_click to [row, col] order for comparison:
manual_point = np.array([manual_click[1], manual_click[0]])

# Find the index in the contour closest to the manual start point.
start_idx = np.argmin(np.linalg.norm(selected_contour - manual_point, axis=1))

# "Roll" the contour so that it starts at the manual start point.
ordered_contour = np.roll(selected_contour, -start_idx, axis=0)

# Now, we want the contour segment from the manual start to the detected end.
# Here, we define the detected end as the point in the ordered contour with the maximum y-value (i.e., maximum row).
end_idx = np.argmax(ordered_contour[:, 0])
contour_segment = ordered_contour[:end_idx+1]

# For visualization, mark the manual start (green) and detected end (red).
plt.figure(figsize=(roi.shape[1]/100, roi.shape[0]/100))
plt.imshow(roi, interpolation='nearest', cmap='gray')
plt.plot(contour_segment[:, 1], contour_segment[:, 0], 'b-', label="Yarn Contour")
plt.plot(manual_click[0], manual_click[1], 'go', markersize=5, label="Manual Start")
plt.plot(contour_segment[-1, 1], contour_segment[-1, 0], 'ro', markersize=5, label="Detected End")
plt.title("Yarn Contour Segment (from Manual Start to Detected End)")
plt.legend()
plt.show()

# ---------------------------
# Part 3: Equidistant Reparameterization of the Contour Segment
# ---------------------------
# Compute cumulative arc-length along the contour segment.
diffs = np.diff(contour_segment, axis=0)
seg_lengths = np.sqrt((diffs**2).sum(axis=1))
arc_length = np.insert(np.cumsum(seg_lengths), 0, 0)

# Set desired spacing along the arc.
# For a desired spacing of 0.1 cm, convert to pixels using your conversion factor.
# Here, we need to set a desired spacing in pixels. For example, if 1 pixel = 0.01 cm, then 0.1 cm = 10 pixels.
# (You may adjust PIXEL_TO_CM accordingly.)
PIXEL_TO_CM = 0.004387504388
desired_spacing_cm = 0.1
s_step_pixels = desired_spacing_cm / PIXEL_TO_CM  # e.g., if PIXEL_TO_CM=0.01, then s_step_pixels = 10

# Generate new arc-length positions at equal intervals.
s_new = np.arange(0, arc_length[-1], s_step_pixels)

# Interpolate x (column) and y (row) coordinates along the arc.
interp_row = interp1d(arc_length, contour_segment[:, 0], kind='linear')
interp_col = interp1d(arc_length, contour_segment[:, 1], kind='linear')
new_rows = interp_row(s_new)
new_cols = interp_col(s_new)
equidistant_points = np.vstack((new_rows, new_cols)).T

# Visualize the equidistant points on the contour segment.
plt.figure(figsize=(roi.shape[1]/100, roi.shape[0]/100))
plt.imshow(roi, interpolation='nearest', cmap='gray')
plt.plot(contour_segment[:, 1], contour_segment[:, 0], 'b-', label="Contour Segment")
plt.plot(equidistant_points[:, 1], equidistant_points[:, 0], 'ro', markersize=2, label="Equidistant Points")
plt.title("Equidistant Points along Yarn Contour Segment")
plt.legend()
plt.show()

# ---------------------------
# Part 4: Compute Downward Displacement
# ---------------------------
# Define downward displacement as the difference in y-coordinate (row value)
# between the first and last equidistant points.
downward_disp_pixels = equidistant_points[-1, 0] - equidistant_points[0, 0]
print(f"Downward Displacement (pixels): {downward_disp_pixels:.2f}")

# Convert displacement to centimeters using a preset conversion factor.
# (PIXEL_TO_CM here is assumed; adjust based on calibration.)
PIXEL_TO_CM = 0.01  # e.g., 1 pixel = 0.01 cm
downward_disp_cm = downward_disp_pixels * PIXEL_TO_CM
print(f"Downward Displacement (cm): {downward_disp_cm:.4f} cm")

# Assume equidistant_points has been computed in previous steps.
# If you need them as a list, you can also do:
equidistant_points_list = equidistant_points.tolist()

# For clarity, remember that:
#   equidistant_points[:, 0] -> y-coordinates (rows)
#   equidistant_points[:, 1] -> x-coordinates (columns)

# --- Cubic Polynomial Fitting ---
# Fit a cubic polynomial (degree 3) using np.polyfit:
coeffs = np.polyfit(equidistant_points[:, 1], equidistant_points[:, 0], 3)
poly_func = np.poly1d(coeffs)

# Create a fine set of x values over the range of your data:
x_fit = np.linspace(equidistant_points[:, 1].min(), equidistant_points[:, 1].max(), 200)
y_fit = poly_func(x_fit)

# Optionally, compute R² for the fit:
from sklearn.metrics import r2_score
y_pred = poly_func(equidistant_points[:, 1])
r2 = r2_score(equidistant_points[:, 0], y_pred)
print("Cubic Fit Coefficients:", coeffs)
print(f"R²: {r2:.4f}")

# --- Visualization: Overlay the fitted curve on the original ROI ---
plt.figure(figsize=(roi.shape[1]/100, roi.shape[0]/100))
plt.imshow(roi, interpolation='nearest')
plt.plot(equidistant_points[:, 1], equidistant_points[:, 0], 'ro', markersize=2, label="Equidistant Points")
plt.plot(x_fit, y_fit, 'g-', linewidth=2, label="Cubic Polynomial Fit")
plt.title("Cubic Polynomial Fit on Equidistant Points")
plt.legend()
plt.show()
