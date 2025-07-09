import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

# --- 1. ArUco Board and Printer Settings ---

# Describe the board you want to create
ARUCO_DICT_NAME = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.036  # meters
MARKER_SEPARATION = 0.01   # meters 
BOARD_ROWS = 4
BOARD_COLS = 6

# Define the physical paper size
PAPER_WIDTH_IN = 11.0
PAPER_HEIGHT_IN = 8.5

# Safe margin for printing in inches
MARGIN_IN = 0.25

# High DPI for good print quality
DPI = 600


# --- 2. Calculations ---

# Create the digital blueprint of the board
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
board = cv2.aruco.GridBoard((BOARD_COLS, BOARD_ROWS), MARKER_LENGTH, MARKER_SEPARATION, aruco_dict, firstMarkerID=0)

# Calculate the board's real-world dimensions in meters and inches
m_to_in = 39.3701
board_width_m = (BOARD_COLS * MARKER_LENGTH) + ((BOARD_COLS - 1) * MARKER_SEPARATION)
board_height_m = (BOARD_ROWS * MARKER_LENGTH) + ((BOARD_ROWS - 1) * MARKER_SEPARATION)
board_width_in = board_width_m * m_to_in
board_height_in = board_height_m * m_to_in

# Calculate the safe content area on the paper
content_width_safe = PAPER_WIDTH_IN - (2 * MARGIN_IN)
content_height_safe = PAPER_HEIGHT_IN - (2 * MARGIN_IN)

# Check if the board fits within the safe area
if board_width_in > content_width_safe or board_height_in > content_height_safe:
    print("❌ ERROR: The board is too large for the paper with the specified margins.")
    print(f"    Board size: {board_width_in:.2f} x {board_height_in:.2f} inches")
    print(f"    Safe area: {content_width_safe:.2f} x {content_height_safe:.2f} inches")
    exit()

# Calculate the pixel dimensions for the final image to be generated
img_width_px = int(content_width_safe * DPI)
img_height_px = int(content_height_safe * DPI)


# --- 3. Generate and Save PDF ---

# Generate the board image to fit the safe content area
# Note: The board will be centered within this image with whitespace around it.
board_image = board.generateImage((img_width_px, img_height_px), marginSize=20, borderBits=1)

# Setup for saving
parser = argparse.ArgumentParser(description='Generate a printable ArUco board with safe margins.')
parser.add_argument('--output', type=str, default='printable_aruco_board.pdf')
args = parser.parse_args()
pdf_path = args.output

# Create a PDF page with the exact physical dimensions of the paper
with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, PAPER_HEIGHT_IN), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]) # Use the entire figure space
    ax.imshow(board_image, cmap='gray')
    ax.axis('off')

    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print(f"✅ PDF successfully created: {pdf_path}")
print(f"   Board physical size: {board_width_in:.2f} x {board_height_in:.2f} inches")
print("\n🖨️ IMPORTANT: Print the PDF using your printer's 'Actual Size' or '100% Scale' setting.")