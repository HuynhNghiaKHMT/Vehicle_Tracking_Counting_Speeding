import pandas as pd
import datetime
import os

class VehicleDataManager:
    def __init__(self, excel_filename="vehicle_information.xlsx"):
        self.excel_filename = excel_filename
        self.records = [] # List to store dictionaries of vehicle data
        self.tracked_vehicle_ids = set() # To ensure each vehicle is recorded only once

        # Define the Excel columns
        self.columns = ["Time", "ID", "Class", "Average Speed (km/h)"]

        # Ensure the Excel file and headers exist if it's a new file
        if not os.path.exists(self.excel_filename):
            self._create_empty_excel()

    def _create_empty_excel(self):
        """Creates an empty Excel file with headers."""
        df = pd.DataFrame(columns=self.columns)
        df.to_excel(self.excel_filename, index=False)

    def add_vehicle_record(self, vehicle_id, vehicle_class_history, vehicle_speed_history):
        """
        Adds a record of a vehicle that has crossed the line.
        Ensures each vehicle ID is recorded only once per crossing event.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine the most frequent class (majority voting)
        if vehicle_class_history:
            # Count occurrences of each class
            class_counts = {}
            for cls in vehicle_class_history:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            # Get the class with the highest count
            most_frequent_class = max(class_counts, key=class_counts.get)
        else:
            most_frequent_class = "Unknown"

        # Calculate average speed
        if vehicle_speed_history:
            avg_speed = sum(vehicle_speed_history) / len(vehicle_speed_history)
        else:
            avg_speed = 0.0 # Or np.nan for 'Not a Number'

        record = {
            "Time": current_time,
            "ID": vehicle_id,
            "Majority voting class": most_frequent_class,
            "Average Speed (km/h)": f"{avg_speed:.2f}" 
        }
        self.records.append(record)
        self.tracked_vehicle_ids.add(vehicle_id) # Add to set (optional, for internal tracking if needed)
        print(f"Recorded vehicle ID: {vehicle_id}, Class: {most_frequent_class}, Avg Speed: {avg_speed:.2f} km/h")
        return True # Indicate record was added

    def save_to_excel(self):
        """Appends the accumulated records to the Excel file."""
        if not self.records:
            print("No new records to save to Excel.")
            return

        # Load existing data
        try:
            existing_df = pd.read_excel(self.excel_filename)
        except FileNotFoundError:
            existing_df = pd.DataFrame(columns=self.columns) # Create empty if not found

        new_df = pd.DataFrame(self.records, columns=self.columns)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Clear records after saving
        self.records = []
        self.tracked_vehicle_ids.clear() # Clear internal tracking set too

        # Save to Excel
        try:
            updated_df.to_excel(self.excel_filename, index=False)
            print(f"Successfully saved records to {self.excel_filename}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")