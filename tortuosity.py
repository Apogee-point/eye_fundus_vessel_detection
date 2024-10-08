import cv2  # type: ignore
import numpy as np  # type: ignore
import time

# Load the preprocessed vessel image (binarized or segmented)
vessel_image = cv2.imread("./output/output.jpg", 0)

# Detect contours in the vessel image
contours, _ = cv2.findContours(vessel_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
output_image = cv2.cvtColor(vessel_image, cv2.COLOR_GRAY2BGR)

cv2.imshow("Original Vessel Image", vessel_image)
cv2.waitKey(0)

tortuosity_values = []

for cnt in contours:
    # Compute the curvilinear (arc) length of the vessel segment
    arc_len = cv2.arcLength(cnt, True)

    # Find the endpoints of the vessel segment (using the most distant points)
    dist = cv2.pointPolygonTest(cnt, (0, 0), True)
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])

    # Compute the Euclidean distance (straight line) between the endpoints
    straight_len = np.sqrt(
        (leftmost[0] - rightmost[0]) ** 2 + (leftmost[1] - rightmost[1]) ** 2
    )

    if straight_len != 0:  # Avoid division by zero
        # Tortuosity is the ratio of curvilinear length to straight-line distance
        tortuosity = arc_len / straight_len
        tortuosity_values.append(tortuosity)
        print(f"Tortuosity of vessel segment: {tortuosity:.2f}")

        # Color the vessel segment based on tortuosity value
        # Blue for low tortuosity, Red for high tortuosity
        color = (0, 0, int(min(255, tortuosity * 20)))  # Scale tortuosity to a color
        cv2.drawContours(output_image, [cnt], -1, color, 2)

if tortuosity_values:
    avg_tortuosity = np.mean(tortuosity_values)
    print(f"Average tortuosity of the vessel network: {avg_tortuosity:.2f}")
else:
    print("No tortuosity values were calculated.")

cv2.imshow("Tortuosity Heatmap", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
