#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

# --- 1. ArUco Board and Printer Settings ---

ARUCO_DICT_NAME = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.036      # meters
MARKER_SEPARATION = 0.01   # meters
BOARD_ROWS = 4
BOARD_COLS = 6

PAPER_WIDTH_IN = 11.0
PAPER_HEIGHT_IN = 8.5
DPI = 600

# Desired compensation factor for printer shrinkage
DESIRED_SCALE_FACTOR = 1.06   # adjust here if needed

m_to_in = 39.3701
inch_per_meter = m_to_in
pixels_per_inch = DPI
pixels_per_meter = inch_per_meter * pixels_per_inch

# --- 2. Calculations ---

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
board = cv2.aruco.GridBoard(
    (BOARD_COLS, BOARD_ROWS),
    MARKER_LENGTH,
    MARKER_SEPARATION,
    aruco_dict
)

# Real board size in meters
board_width_m = (BOARD_COLS * MARKER_LENGTH) + ((BOARD_COLS - 1) * MARKER_SEPARATION)
board_height_m = (BOARD_ROWS * MARKER_LENGTH) + ((BOARD_ROWS - 1) * MARKER_SEPARATION)

# Convert to inches
board_width_in = board_width_m * m_to_in
board_height_in = board_height_m * m_to_in

# Use full page area
content_width_in = PAPER_WIDTH_IN
content_height_in = PAPER_HEIGHT_IN
content_width_px = int(content_width_in * DPI)
content_height_px = int(content_height_in * DPI)

# Compute maximum allowed scale factor
max_scale_width = content_width_in / board_width_in
max_scale_height = content_height_in / board_height_in
max_allowed_scale = min(max_scale_width, max_scale_height)

# Final scale factor: minimum of desired and maximum allowed
FINAL_SCALE_FACTOR = min(DESIRED_SCALE_FACTOR, max_allowed_scale)

if FINAL_SCALE_FACTOR < DESIRED_SCALE_FACTOR:
    print(f"⚠️ WARNING: Desired scale factor ({DESIRED_SCALE_FACTOR:.3f}) too large to fit.")
    print(f"Using maximum possible scale factor: {FINAL_SCALE_FACTOR:.3f}")
else:
    print(f"✅ Using desired scale factor: {FINAL_SCALE_FACTOR:.3f}")

# Board size in pixels, scaled
board_width_px = int(board_width_m * pixels_per_meter * FINAL_SCALE_FACTOR)
board_height_px = int(board_height_m * pixels_per_meter * FINAL_SCALE_FACTOR)

# --- 3. Generate Board Image ---

board_image = board.generateImage(
    (board_width_px, board_height_px),
    marginSize=0,
    borderBits=1
)

# Create a blank page canvas and center the board on it
canvas = np.ones((content_height_px, content_width_px), dtype=np.uint8) * 255
y_offset = (content_height_px - board_height_px) // 2
x_offset = (content_width_px - board_width_px) // 2
canvas[y_offset:y_offset+board_height_px, x_offset:x_offset+board_width_px] = board_image

# --- 4. Save to PDF ---

parser = argparse.ArgumentParser(description='Generate a printable ArUco board.')
parser.add_argument('--output', type=str, default='printable_aruco_board.pdf')
args = parser.parse_args()
pdf_path = args.output

with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, PAPER_HEIGHT_IN), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, cmap='gray', extent=[0, PAPER_WIDTH_IN, 0, PAPER_HEIGHT_IN])
    ax.axis('off')

    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print(f"✅ PDF successfully created: {pdf_path}")
print(f"   Logical board size: {board_width_in:.2f} x {board_height_in:.2f} inches")
print(f"   Printed board scaled by {FINAL_SCALE_FACTOR*100:.1f}% to compensate printer shrinkage.")
print("\n🖨️ Print the PDF using your printer's 'Actual Size' or '100% Scale' setting.")
