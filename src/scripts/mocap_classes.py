import csv
import os
import numpy as np

class Marker:
    """
    A class to represent a single marker from motion capture data.
    Each coordinate (x, y, z) is stored as a NumPy array.

    Attributes:
        name (str): The name of the marker (e.g., 'marker1').
        x (np.ndarray): A NumPy array of all x-coordinates for each frame.
        y (np.ndarray): A NumPy array of all y-coordinates for each frame.
        z (np.ndarray): A NumPy array of all z-coordinates for each frame.
    """
    def __init__(self, name):
        """Initializes the Marker object with temporary lists for data collection."""
        self.name = name
        # Data is collected in lists for performance, then converted to NumPy arrays.
        self._x_list = []
        self._y_list = []
        self._z_list = []
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])

    def add_frame_data(self, position):
        """
        Appends a single frame of position data to the internal lists.

        Args:
            position (dict): A dictionary representing the 3D position
                             (e.g., {'x': 100.5, 'y': 200.2, 'z': None}).
        """
        self._x_list.append(position.get('x'))
        self._y_list.append(position.get('y'))
        self._z_list.append(position.get('z'))

    def finalize_data(self):
        """
        Converts the internal lists of coordinates to NumPy arrays.
        This should be called after all data has been added.
        """
        # dtype=float automatically converts None to np.nan
        self.x = np.array(self._x_list, dtype=float)
        self.y = np.array(self._y_list, dtype=float)
        self.z = np.array(self._z_list, dtype=float)
        # Clear the lists to save memory
        self._x_list = []
        self._y_list = []
        self._z_list = []


    def __repr__(self):
        """Provides a developer-friendly string representation of the object."""
        return f"Marker(name='{self.name}', frames_captured={self.get_frame_count()})"

    # --- GETTER METHODS ---

    def get_frame_count(self):
        """Returns the total number of captured frames."""
        return len(self.x)

    def get_position_at_frame(self, frame_number):
        """
        Retrieves the x, y, z position for a specific frame number.

        Args:
            frame_number (int): The frame to retrieve (1-based index).

        Returns:
            dict: A dictionary with x, y, z keys, or None if frame is out of bounds.
        """
        # Adjust for 0-based list index
        index = frame_number - 1
        if 0 <= index < self.get_frame_count():
            return {
                'x': self.x[index],
                'y': self.y[index],
                'z': self.z[index]
            }
        return None # Frame number is not valid

    def get_all_x_positions(self):
        """Returns the NumPy array of all x-position coordinates."""
        return self.x

    def get_all_y_positions(self):
        """Returns the NumPy array of all y-position coordinates."""
        return self.y

    def get_all_z_positions(self):
        """Returns the NumPy array of all z-position coordinates."""
        return self.z
        
    def get_all_positions_as_array(self):
        """
        Returns all position data as a single (N, 3) NumPy array.
        This is ideal for visualization and ML tasks.
        """
        return np.stack((self.x, self.y, self.z), axis=-1)


def extract_marker_data(filepath):
    """
    Parses a motion capture CSV file with multi-level headers to extract data
    for all markers.
    """
    markers = {}
    marker_column_map = {}

    try:
        with open(filepath, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            all_rows = list(reader)

        # --- 1. Find the start of the data and the header block ---
        data_start_index = -1
        for i, row in enumerate(all_rows):
            if row and row[0].strip().isdigit():
                data_start_index = i
                break
        
        if data_start_index == -1:
            raise ValueError("Could not find the start of data (a row beginning with a frame number).")

        header_block = all_rows[:data_start_index]
        data_rows = all_rows[data_start_index:]

        # --- 2. Robustly find header rows by independently searching for their content ---
        name_row, property_row, axis_row = None, None, None
        
        for row in header_block:
            row_content = set(cell.strip() for cell in row)
            if 'Name' in row_content:
                name_row = row
            if 'Position' in row_content:
                property_row = row
            if 'X' in row_content and 'Y' in row_content and 'Z' in row_content:
                axis_row = row

        if not name_row and property_row:
            prop_index = header_block.index(property_row)
            if prop_index > 0:
                name_row = header_block[prop_index - 1]

        if not all([name_row, property_row, axis_row]):
            raise ValueError("Could not find all required header rows ('Name', 'Property', 'Axis'). Please check the CSV format.")

        num_columns = len(axis_row)
        for r in [name_row, property_row, axis_row]:
            while len(r) < num_columns: r.append('')

        # --- 3. Map all markers to their column indices ---
        for i in range(num_columns):
            asset_name = name_row[i].strip()
            property_type = property_row[i].strip()
            axis = axis_row[i].strip().upper()

            if property_type == 'Position' and axis in ['X', 'Y', 'Z']:
                if asset_name not in markers:
                    markers[asset_name] = Marker(name=asset_name)
                    marker_column_map[asset_name] = {}
                
                marker_column_map[asset_name][axis] = i

        # --- 4. Process the data rows using the generated index map ---
        for row in data_rows:
            if not row or len(row) < num_columns: continue
            
            for name, mapping in marker_column_map.items():
                try:
                    if not all(k in mapping for k in ['X', 'Y', 'Z']):
                        continue
                    
                    x_val = row[mapping['X']].strip()
                    y_val = row[mapping['Y']].strip()
                    z_val = row[mapping['Z']].strip()

                    position = {
                        'x': float(x_val) if x_val else None,
                        'y': float(y_val) if y_val else None,
                        'z': float(z_val) if z_val else None,
                    }
                    markers[name].add_frame_data(position)
                except (ValueError, IndexError, KeyError):
                    continue
        
        # --- 5. Finalize data by converting lists to NumPy arrays ---
        for marker in markers.values():
            marker.finalize_data()

        final_markers = {name: marker for name, marker in markers.items() if marker.get_frame_count() > 0}

        return final_markers

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    # Point the script to your new CSV file
    csv_file_path = '/Users/manasj326/Desktop/extract/Take 2025-06-30 10.00.02 AM.csv'

    # Set NumPy print options to show the full arrays without truncation
    np.set_printoptions(threshold=np.inf)

    if not os.path.exists(csv_file_path):
        print(f"File not found: '{csv_file_path}'")
        print("Please ensure the CSV file is in the same directory as this script.")
    else:
        # The main function that parses the file and returns a dictionary of Marker objects
        extracted_data = extract_marker_data(csv_file_path)

        if extracted_data:
            print(f"✅ Successfully extracted data for {len(extracted_data)} marker(s).")

            # --- Main function to get specific data ---
            # Define a list of markers you want to inspect
            target_marker_names = ['marker1', 'marker2']

            # Loop through the list of target names
            for target_name in target_marker_names:
                # Check if the desired marker was found in the file
                if target_name in extracted_data:
                    # Get the specific Marker object from the dictionary
                    target_marker = extracted_data[target_name]

                    # Use the getter method to retrieve only the x-positions
                    x_positions = target_marker.get_all_x_positions()

                    print("\n" + "="*40)
                    print(f"Retrieving X-Positions for: '{target_name}'")
                    print("="*40)
                    print(x_positions)
                else:
                    print(f"\n❌ Could not find the marker named '{target_name}' in the file.")
            
            # You can still print the available markers if you need to see all of them
            print(f"\nℹ️ Available markers in the file are: {list(extracted_data.keys())}")

        else:
            print("❌ Could not extract any marker data.")
