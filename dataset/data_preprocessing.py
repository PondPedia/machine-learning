import pandas as pd
import numpy as np

# Select which pond that you want to clean
pond = 10

# Import the CSV file
df = pd.read_csv(f"./raw/IoTPond{pond}.csv")

#  format="%Y-%m-%d %H:%M:%S %Z"

# Change the first column of data (Data) to datetype type
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, format='mixed')

# Select column that contain specific words
temp = df.columns.str.contains('temp', case=False)
oxygen = df.columns.str.contains('oxygen', case=False)
ph = df.columns.str.contains('ph', case=False)
ammonia = df.columns.str.contains('ammonia', case=False)
nitrat = df.columns.str.contains('nitrat', case=False)

# Get the index of the column above
temp_indices = [index for index, i in enumerate(temp) if i == True][0]
ph_indices = [index for index, i in enumerate(ph) if i == True][0]
oxygen_indices = [index for index, i in enumerate(oxygen) if i == True][0]
ammonia_indices = [index for index, i in enumerate(ammonia) if i == True][0]
nitrat_indices = [index for index, i in enumerate(nitrat) if i == True][0]

# Keep the values in the "Temperature (C)" column that are between 20 and 35, and replace the values outside the range with 0
df.iloc[:, temp_indices] = np.where(df.iloc[:, temp_indices].between(20, 35), df.iloc[:, temp_indices], 0)

# Keep the values in the "Dissolved Oxygen(g/ml)" column that are between 0 and 10, and replace the values outside the range with 0
df.iloc[:, oxygen_indices] = np.where(df.iloc[:, oxygen_indices].between(0, 10), df.iloc[:, oxygen_indices], 0)

# Keep the values in the "pH" column that are between 1 and 14, and replace the values outside the range with 0
df.iloc[:, ph_indices] = np.where(df.iloc[:, ph_indices].between(1, 14), df.iloc[:, ph_indices], 0)

# Keep the values in the "ammonia" column that are between 0 and 1, and replace the values outside the range with 0
df.iloc[:, ammonia_indices] = np.where(df.iloc[:, ammonia_indices].between(0, 1), df.iloc[:, ammonia_indices], 0)

# Keep the values in the "nitrat" column that are between 0 and 20, and replace the values outside the range with 0
df.iloc[:, nitrat_indices] = np.where(df.iloc[:, nitrat_indices].between(0, 20), df.iloc[:, nitrat_indices], 0)

# Save the cleaned CSV file
df.to_csv(f"./processed/cleaned_IoTPond{pond}.csv", index=False)


print('DONE')
