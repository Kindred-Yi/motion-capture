import csv
import os
import numpy as np

class RigidBody:
    """
    A class to represent a rigid body from motion capture data.
    Each coordinate is stored as a NumPy array.

    Attributes:
        name (str): The name of the rigid body.
        pos_x, pos_y, pos_z (np.ndarray): NumPy arrays of position coordinates.
        rot_w, rot_x, rot_y, rot_z (np.ndarray): NumPy arrays of rotation coordinates.
    """
    def __init__(self, name):
        """Initializes the RigidBody object with temporary lists."""
        self.name = name
        # Use lists for efficient appending during parsing
        self._pos_x_list, self._pos_y_list, self._pos_z_list = [], [], []
        self._rot_w_list, self._rot_x_list, self._rot_y_list, self._rot_z_list = [], [], [], []
        # Final data will be stored in NumPy arrays
        self.pos_x, self.pos_y, self.pos_z = np.array([]), np.array([]), np.array([])
        self.rot_w, self.rot_x, self.rot_y, self.rot_z = np.array([]), np.array([]), np.array([]), np.array([])


    def add_frame_data(self, position, rotation):
        """
        Appends a single frame of data to the internal coordinate lists.

        Args:
            position (dict): A dictionary for 3D position {'x', 'y', 'z'}.
            rotation (dict): A dictionary for rotation {'w', 'x', 'y', 'z'}.
        """
        self._pos_x_list.append(position.get('x'))
        self._pos_y_list.append(position.get('y'))
        self._pos_z_list.append(position.get('z'))
        self._rot_w_list.append(rotation.get('w'))
        self._rot_x_list.append(rotation.get('x'))
        self._rot_y_list.append(rotation.get('y'))
        self._rot_z_list.append(rotation.get('z'))

    def finalize_data(self):
        """
        Converts the internal lists of coordinates to NumPy arrays.
        This should be called after all data has been added.
        """
        # dtype=float automatically converts None to np.nan
        self.pos_x = np.array(self._pos_x_list, dtype=float)
        self.pos_y = np.array(self._pos_y_list, dtype=float)
        self.pos_z = np.array(self._pos_z_list, dtype=float)
        self.rot_w = np.array(self._rot_w_list, dtype=float)
        self.rot_x = np.array(self._rot_x_list, dtype=float)
        self.rot_y = np.array(self._rot_y_list, dtype=float)
        self.rot_z = np.array(self._rot_z_list, dtype=float)
        # Clear the lists to save memory
        self._pos_x_list, self._pos_y_list, self._pos_z_list = [], [], []
        self._rot_w_list, self._rot_x_list, self._rot_y_list, self._rot_z_list = [], [], [], []


    def __repr__(self):
        """Provides a developer-friendly string representation of the object."""
        return f"RigidBody(name='{self.name}', frames_captured={self.get_frame_count()})"

    # --- GETTER METHODS ---

    def get_frame_count(self):
        """Returns the total number of captured frames."""
        return len(self.pos_x)

    def get_position_at_frame(self, frame_number):
        """Retrieves the x, y, z position for a specific frame number."""
        index = frame_number - 1
        if 0 <= index < self.get_frame_count():
            return {'x': self.pos_x[index], 'y': self.pos_y[index], 'z': self.pos_z[index]}
        return None

    def get_rotation_at_frame(self, frame_number):
        """Retrieves the w, x, y, z rotation for a specific frame number."""
        index = frame_number - 1
        if 0 <= index < self.get_frame_count():
            return {'w': self.rot_w[index], 'x': self.rot_x[index], 'y': self.rot_y[index], 'z': self.rot_z[index]}
        return None

    def get_all_x_positions(self):
        """Returns the NumPy array of all x-position coordinates."""
        return self.pos_x

    def get_all_y_positions(self):
        """Returns the NumPy array of all y-position coordinates."""
        return self.pos_y

    def get_all_z_positions(self):
        """Returns the NumPy array of all z-position coordinates."""
        return self.pos_z

    def get_all_w_rotations(self):
        """Returns the NumPy array of all w-rotation coordinates."""
        return self.rot_w

    def get_all_x_rotations(self):
        """Returns the NumPy array of all x-rotation coordinates."""
        return self.rot_x

    def get_all_y_rotations(self):
        """Returns the NumPy array of all y-rotation coordinates."""
        return self.rot_y

    def get_all_z_rotations(self):
        """Returns the NumPy array of all z-rotation coordinates."""
        return self.rot_z
        
    def get_all_positions_as_array(self):
        """
        Returns all position data as a single (N, 3) NumPy array.
        This is ideal for visualization and ML tasks.
        """
        return np.stack((self.pos_x, self.pos_y, self.pos_z), axis=-1)

    def get_all_rotations_as_array(self):
        """
        Returns all rotation data as a single (N, 4) NumPy array.
        This is ideal for visualization and ML tasks.
        """
        return np.stack((self.rot_w, self.rot_x, self.rot_y, self.rot_z), axis=-1)


def extract_rigid_body_data(filepath):
    """
    Parses a motion capture CSV file with multi-level headers to extract data
    for all rigid bodies.
    """
    rigid_bodies = {}
    rb_column_map = {}

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
        type_row, name_row, property_row, axis_row = None, None, None, None
        
        for row in header_block:
            row_content = set(cell.strip() for cell in row)
            if 'Type' in row_content: type_row = row
            if 'Name' in row_content: name_row = row
            if 'Position' in row_content or 'Rotation' in row_content: property_row = row
            if 'X' in row_content and 'Y' in row_content and 'Z' in row_content: axis_row = row

        if not all([type_row, name_row, property_row, axis_row]):
            raise ValueError("Could not find all required header rows ('Type', 'Name', 'Property', 'Axis').")

        num_columns = len(axis_row)
        for r in [type_row, name_row, property_row, axis_row]:
            while len(r) < num_columns: r.append('')

        # --- 3. Map all assets to their column indices ---
        for i in range(num_columns):
            asset_type = type_row[i].strip()
            asset_name = name_row[i].strip()
            property_type = property_row[i].strip()
            axis = axis_row[i].strip().upper()

            if not asset_type or not asset_name or not axis: continue

            if asset_type == 'Rigid Body':
                if asset_name not in rigid_bodies:
                    rigid_bodies[asset_name] = RigidBody(name=asset_name)
                    rb_column_map[asset_name] = {'Position': {}, 'Rotation': {}}
                if property_type in rb_column_map[asset_name]:
                    rb_column_map[asset_name][property_type][axis] = i

        # --- 4. Process the data rows using the generated index maps ---
        for row in data_rows:
            if not row or len(row) < num_columns: continue
            
            for name, mapping in rb_column_map.items():
                try:
                    pos_map = mapping.get('Position', {})
                    rot_map = mapping.get('Rotation', {})
                    if not all(k in pos_map for k in ['X', 'Y', 'Z']) or not all(k in rot_map for k in ['W', 'X', 'Y', 'Z']): continue
                    
                    pos_x = row[pos_map['X']].strip()
                    pos_y = row[pos_map['Y']].strip()
                    pos_z = row[pos_map['Z']].strip()
                    rot_w = row[rot_map['W']].strip()
                    rot_x = row[rot_map['X']].strip()
                    rot_y = row[rot_map['Y']].strip()
                    rot_z = row[rot_map['Z']].strip()

                    position = {'x': float(pos_x) if pos_x else None, 'y': float(pos_y) if pos_y else None, 'z': float(pos_z) if pos_z else None}
                    rotation = {'w': float(rot_w) if rot_w else None, 'x': float(rot_x) if rot_x else None, 'y': float(rot_y) if rot_y else None, 'z': float(rot_z) if rot_z else None}
                    
                    rigid_bodies[name].add_frame_data(position, rotation)
                except (ValueError, IndexError, KeyError): continue
        
        # --- 5. Finalize data by converting lists to NumPy arrays ---
        for body in rigid_bodies.values():
            body.finalize_data()

        final_rigid_bodies = {name: body for name, body in rigid_bodies.items() if body.get_frame_count() > 0}

        return final_rigid_bodies

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None

if __name__ == "__main__":
    # Point the script to your new CSV file
    csv_file_path = 'scenario 1 task 1(in).csv'

    # Set NumPy print options to show the full arrays without truncation
    np.set_printoptions(threshold=np.inf)

    if not os.path.exists(csv_file_path):
        print(f"File not found: '{csv_file_path}'")
        print("Please ensure the CSV file is in the same directory as this script.")
    else:
        # The main function that parses the file and returns a dictionary of RigidBody objects
        extracted_data = extract_rigid_body_data(csv_file_path)

        if extracted_data:
            print(f"✅ Successfully extracted data for {len(extracted_data)} rigid body/bodies.")

            # --- Main function to get specific data ---
            # Define a list of rigid bodies you want to inspect
            target_body_names = ['plate', 'peanut butter cap']

            # Loop through the list of target names
            for target_name in target_body_names:
                # Check if the desired rigid body was found in the file
                if target_name in extracted_data:
                    # Get the specific RigidBody object from the dictionary
                    target_body = extracted_data[target_name]

                    # Use the getter method to retrieve only the x-positions
                    x_positions = target_body.get_all_x_positions()

                    print("\n" + "="*40)
                    print(f"Retrieving X-Positions for: '{target_name}'")
                    print("="*40)
                    print(x_positions)
                else:
                    print(f"\n❌ Could not find the rigid body named '{target_name}' in the file.")
            
            # You can still print the available bodies if you need to see all of them
            print(f"\nℹ️ Available bodies in the file are: {list(extracted_data.keys())}")

        else:
            print("❌ Could not extract any rigid body data.")
