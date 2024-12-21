import os
import json
import pandas as pd

# File paths
excel_file = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Dataset\Dataset1\images_info.xlsx"
image_folder = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Dataset\main_DATASET\images"
json_file = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Dataset\Dataset2\captions.json"

output_file = r"C:\Users\thoma\Documents\Thomas - SSD\IA - Image Generator\Dataset\main_DATASET\labels\unified_descriptions.json"

# Load descriptions from Excel
try:
    df = pd.read_excel(excel_file)
    descriptions_from_excel = df.set_index('image')['caption'].to_dict()

except Exception as e:
    raise ValueError(f"Error reading Excel file: {e}")

# Load descriptions from JSON
try:
    with open(json_file, "r") as f:
        descriptions_from_json = json.load(f)

except Exception as e:
    raise ValueError(f"Error reading JSON file: {e}")

# Merge dictionaries (JSON overwrites Excel if keys overlap)
unified_descriptions = {**descriptions_from_excel, **descriptions_from_json}

# Save the unified descriptions
try:
    with open(output_file, "w") as f:
        json.dump(unified_descriptions, f, indent=4)

except Exception as e:
    raise ValueError(f"Error writing output JSON file: {e}")
