import json
import pandas as pd

# ✅ Set the correct path to your JSON file
json_file_path = "dataset/News_Category_Dataset_v2.json"

try:
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    # ✅ Convert JSON data to pandas DataFrame
    df = pd.DataFrame(data)

    # ✅ Save it as a CSV file
    csv_output_path = "dataset/News_Category_Dataset_v2.csv"
    df.to_csv(csv_output_path, index=False)

    print("✅ JSON file successfully converted to CSV and saved as:")
    print(csv_output_path)

except FileNotFoundError:
    print("❌ File not found! Please check the path:")
    print(json_file_path)
