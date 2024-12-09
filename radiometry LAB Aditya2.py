#!/usr/bin/env python
# coding: utf-8

# # 3.1

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the ASD DN spectra and Sphere Radiance data
asd_dn_path = r"C:\Users\adity\Downloads\ASD-DN-spectra.xlsx"
sphere_radiance_path = r"C:\Users\adity\Downloads\sphere-radiance-spectra-interp-corr.xlsx"


asd_dn_df = pd.read_excel(asd_dn_path, sheet_name='Sheet1')
sphere_radiance_df = pd.read_excel(sphere_radiance_path, sheet_name='Sheet1')

# Extract the wavelengths, DN values, and integration times
asd_dn_wavelengths = asd_dn_df.iloc[1:, 0].astype(float)
asd_dn_values = asd_dn_df.iloc[1:, 1:].astype(float)

# Integration times for DNR calculation (as provided by your friend)
integration_times = [68, 68.1, 136, 136.1, 136.2]

# Calculate DNR (DN Rates) by dividing each DN value by the corresponding integration time
dn_rates = asd_dn_values.copy()
for i, time in enumerate(integration_times):
    dn_rates.iloc[:, i] = asd_dn_values.iloc[:, i] / integration_times[i]

# Extract the radiance values (L values) from the sphere radiance data
sphere_radiance_values = sphere_radiance_df.iloc[:, 1:].astype(float)

# Perform linear regression to find the slope (m) and intercept (b) for each wavelength
slopes = []
intercepts = []
for i in range(len(asd_dn_wavelengths)):
    x = dn_rates.iloc[i, :].values.reshape(-1, 1)  # DNR values
    y = sphere_radiance_values.iloc[i, :].values  # Radiance values
    model = LinearRegression().fit(x, y)
    slopes.append(model.coef_[0])
    intercepts.append(model.intercept_)

# Create a DataFrame to store the calibration parameters (slope m and intercept b)
calibration_params = pd.DataFrame({
    "Wavelength (nm)": asd_dn_wavelengths.values,
    "Slope (m)": slopes,
    "Intercept (b)": intercepts
})

# Display the first few rows of the calibration parameters
print(calibration_params.head())

# Plot the DNR spectra (DN Rates)
plt.figure(figsize=(10, 6))
for i in range(dn_rates.shape[1]):
    plt.plot(asd_dn_wavelengths, dn_rates.iloc[:, i], label=f'Light Level {i+1}')
plt.title('DNR Spectra (DN Rates)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('DNR (DN/ms)')
plt.legend()
plt.grid()
plt.show()

# Extract the wavelengths corresponding to the calibrated radiance data
sphere_wavelengths = sphere_radiance_df.iloc[:, 0]  # Assuming first column is wavelength values

# Plot the calibrated radiance spectra (L values)
plt.figure(figsize=(10, 6))
for i in range(sphere_radiance_values.shape[1]):
    plt.plot(sphere_wavelengths, sphere_radiance_values.iloc[:, i], label=f'Light Level {i+1}')
plt.title('Calibrated Radiance Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Radiance (W/m²/µm/sr)')
plt.legend()
plt.grid()
plt.show()


# Plot the slope (m) vs. Wavelength
plt.figure(figsize=(10, 6))
plt.plot(calibration_params["Wavelength (nm)"], calibration_params["Slope (m)"], label='Slope (m)')
plt.title('Slope (m) vs Wavelength')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Slope (m)')
plt.grid()
plt.legend()
plt.show()

# Plot the intercept (b) vs. Wavelength
plt.figure(figsize=(10, 6))
plt.plot(calibration_params["Wavelength (nm)"], calibration_params["Intercept (b)"], label='Intercept (b)', color='orange')
plt.title('Intercept (b) vs Wavelength')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intercept (b)')
plt.grid()
plt.legend()
plt.show()


# # 3.2

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt

# File paths for the Spectralon data
spectralon_30_path = r"C:\Users\adity\Downloads\GRITT-Spectralon-panel-radiance-i30.csv"
spectralon_60_path = r"C:\Users\adity\Downloads\GRITT-Spectralon-panel-radiance-i60.csv"

# Load the data
spectralon_30_df = pd.read_csv(spectralon_30_path)
spectralon_60_df = pd.read_csv(spectralon_60_path)

# Extract wavelength columns
wavelengths = [float(col) for col in spectralon_30_df.columns[4:]]  # Wavelength range starts from 4th column

# Select 10 view zenith angles (sampled across the range)
view_angles_30 = spectralon_30_df['e: View Zenith Angle'].unique()[:10]
view_angles_60 = spectralon_60_df['e: View Zenith Angle'].unique()[:10]

# Filter the data for these angles
spectralon_30_filtered = spectralon_30_df[spectralon_30_df['e: View Zenith Angle'].isin(view_angles_30)]
spectralon_60_filtered = spectralon_60_df[spectralon_60_df['e: View Zenith Angle'].isin(view_angles_60)]

# Plot spectral radiance for θi = 30°
plt.figure(figsize=(10, 6))
for angle in view_angles_30:
    radiance = spectralon_30_filtered[spectralon_30_filtered['e: View Zenith Angle'] == angle].iloc[:, 4:].mean()
    plt.plot(wavelengths, radiance, label=f"View Angle {angle:.2f}°")
plt.title("Spectral Radiance for θi = 30°")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Radiance (W/m²/µm/sr)")
plt.legend()
plt.grid()
plt.show()

# Plot spectral radiance for θi = 60°
plt.figure(figsize=(10, 6))
for angle in view_angles_60:
    radiance = spectralon_60_filtered[spectralon_60_filtered['e: View Zenith Angle'] == angle].iloc[:, 4:].mean()
    plt.plot(wavelengths, radiance, label=f"View Angle {angle:.2f}°")
plt.title("Spectral Radiance for θi = 60°")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Radiance (W/m²/µm/sr)")
plt.legend()
plt.grid()
plt.show()

# Analyze Lambertian behavior
print("Analysis of Lambertian behavior:")
print("If Spectralon is Lambertian, the radiance should be isotropic (equal in all directions).")
print("You can observe the variation of radiance across view angles from the plots.")


# In[14]:


# Compute average spectral radiance for each wavelength
avg_radiance_30 = spectralon_30_df.iloc[:, 4:].mean(axis=0)
avg_radiance_60 = spectralon_60_df.iloc[:, 4:].mean(axis=0)

# Compute the average of the two L_id(λ) quantities
avg_radiance_combined = (avg_radiance_30 + avg_radiance_60) / 2

# Plot L_id(λ) for θi = 30°, θi = 60°, and their average
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, avg_radiance_30, label="L_id(λ) for θi = 30°")
plt.plot(wavelengths, avg_radiance_60, label="L_id(λ) for θi = 60°")
plt.plot(wavelengths, avg_radiance_combined, label="Average L_id(λ)", linestyle="--")
plt.title("Spectral Radiance and Average")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Radiance (W/m²/µm/sr)")
plt.legend()
plt.grid()
plt.show()


# # 3.3

# In[32]:


import pandas as pd
import matplotlib.pyplot as plt

# File paths for olivine sand data
olivine_63_30_path = r"C:\Users\adity\Downloads\GRITT-olivine-63umto300um-i30-radiance-spectra.csv"
olivine_63_60_path = r"C:\Users\adity\Downloads\GRITT-olivine-63umto300um-i60-radiance-spectra.csv"
olivine_600_30_path = r"C:\Users\adity\Downloads\GRITT-olivine-600umto1000um-i30-radiance-spectra.csv"
olivine_600_60_path = r"C:\Users\adity\Downloads\GRITT-olivine-600umto1000um-i60-radiance-spectra.csv"

# File paths for Spectralon reference data
spectralon_30_path = r"C:\Users\adity\Downloads\GRITT-Spectralon-panel-radiance-i30.csv"
spectralon_60_path = r"C:\Users\adity\Downloads\GRITT-Spectralon-panel-radiance-i60.csv"

# Load Spectralon data
spectralon_30 = pd.read_csv(spectralon_30_path)
spectralon_60 = pd.read_csv(spectralon_60_path)

# Load average L_id(λ) from Spectralon (computed in 3.2)
# Replace with your avg_radiance_combined computation from 3.2
spectralon_avg_radiance = avg_radiance_combined.values

# Load olivine data
olivine_63_30 = pd.read_csv(olivine_63_30_path)
olivine_63_60 = pd.read_csv(olivine_63_60_path)
olivine_600_30 = pd.read_csv(olivine_600_30_path)
olivine_600_60 = pd.read_csv(olivine_600_60_path)

# Extract wavelengths and radiances
wavelengths = [float(col) for col in olivine_63_30.columns[4:]]
olivine_63_30_radiance = olivine_63_30.iloc[:, 4:]
olivine_63_60_radiance = olivine_63_60.iloc[:, 4:]
olivine_600_30_radiance = olivine_600_30.iloc[:, 4:]
olivine_600_60_radiance = olivine_600_60.iloc[:, 4:]

# Extract wavelengths from the Spectralon data
spectralon_wavelengths = [float(col) for col in spectralon_30.columns[4:]]

# Convert spectralon_avg_radiance to a Pandas Series with its original wavelengths as the index
spectralon_avg_radiance_series = pd.Series(spectralon_avg_radiance, index=spectralon_wavelengths)

# Interpolate Spectralon average radiance to match Olivine Sand Radiance wavelengths
spectralon_avg_radiance_aligned = spectralon_avg_radiance_series.reindex(wavelengths, method='nearest')

# Compute BRFs
brf_63_30 = olivine_63_30_radiance / spectralon_avg_radiance_aligned.values
brf_63_60 = olivine_63_60_radiance / spectralon_avg_radiance_aligned.values
brf_600_30 = olivine_600_30_radiance / spectralon_avg_radiance_aligned.values
brf_600_60 = olivine_600_60_radiance / spectralon_avg_radiance_aligned.values

# Set the correct column name for zenith angles
view_angle_column = "e: View Zenith Angle"

# Select 10 view angles (from the original dataset)
selected_angles = olivine_63_30[view_angle_column].unique()[:10]

# Function to plot radiances and BRFs
def plot_radiances_and_brfs(wavelengths, radiance, brf, angles, title_radiance, title_brf):
    plt.figure(figsize=(10, 6))
    for angle in angles:
        index = olivine_63_30[olivine_63_30[view_angle_column] == angle].index[0]
        plt.plot(
            wavelengths,
            radiance.iloc[index, :],
            label=f"View Angle {angle:.2f}°",
        )
    plt.title(title_radiance)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Radiance (W/m²/µm/sr)")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    for angle in angles:
        index = olivine_63_30[olivine_63_30[view_angle_column] == angle].index[0]
        plt.plot(
            wavelengths,
            brf.iloc[index, :],
            label=f"View Angle {angle:.2f}°",
        )
    plt.title(title_brf)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("BRF")
    plt.legend()
    plt.grid()
    plt.show()

# Radiances and BRFs for 63–300 µm
plot_radiances_and_brfs(wavelengths, olivine_63_30_radiance, brf_63_30, selected_angles,
                        "Radiance for 63–300 µm (θi = 30°)", "BRF for 63–300 µm (θi = 30°)")
plot_radiances_and_brfs(wavelengths, olivine_63_60_radiance, brf_63_60, selected_angles,
                        "Radiance for 63–300 µm (θi = 60°)", "BRF for 63–300 µm (θi = 60°)")

# Radiances and BRFs for 600–1000 µm
plot_radiances_and_brfs(wavelengths, olivine_600_30_radiance, brf_600_30, selected_angles,
                        "Radiance for 600–1000 µm (θi = 30°)", "BRF for 600–1000 µm (θi = 30°)")
plot_radiances_and_brfs(wavelengths, olivine_600_60_radiance, brf_600_60, selected_angles,
                        "Radiance for 600–1000 µm (θi = 60°)", "BRF for 600–1000 µm (θi = 60°)")


# # 3.4
# 

# In[31]:


# Normalize BRFs for each configuration
brf_63_30_norm = brf_63_30.div(brf_63_30.max(axis=1), axis=0)
brf_63_60_norm = brf_63_60.div(brf_63_60.max(axis=1), axis=0)
brf_600_30_norm = brf_600_30.div(brf_600_30.max(axis=1), axis=0)
brf_600_60_norm = brf_600_60.div(brf_600_60.max(axis=1), axis=0)

# Function to plot normalized BRFs
def plot_normalized_brfs(wavelengths, brf_norm, angles, title):
    plt.figure(figsize=(10, 6))
    for angle in angles:
        index = olivine_63_30[olivine_63_30[view_angle_column] == angle].index[0]
        plt.plot(
            wavelengths,
            brf_norm.iloc[index, :],
            label=f"View Angle {angle:.2f}°",
        )
    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized BRF")
    plt.legend()
    plt.grid()
    plt.show()

# Select 10 view angles for comparison
selected_angles = olivine_63_30[view_angle_column].unique()[:10]

# Plot normalized BRFs for all configurations
plot_normalized_brfs(wavelengths, brf_63_30_norm, selected_angles, 
                     "Normalized BRF for 63–300 µm (θi = 30°)")
plot_normalized_brfs(wavelengths, brf_63_60_norm, selected_angles, 
                     "Normalized BRF for 63–300 µm (θi = 60°)")
plot_normalized_brfs(wavelengths, brf_600_30_norm, selected_angles, 
                     "Normalized BRF for 600–1000 µm (θi = 30°)")
plot_normalized_brfs(wavelengths, brf_600_60_norm, selected_angles, 
                     "Normalized BRF for 600–1000 µm (θi = 60°)")


# # 3.5

# In[33]:


import numpy as np

# Define step size for integration
delta_theta = np.radians(5)  # Assuming 5-degree steps
cos_theta = np.cos(np.radians(selected_angles))  # Cosine of view zenith angles
sin_theta = np.sin(np.radians(selected_angles))  # Sine of view zenith angles

# Compute DHR for a given normalized BRF DataFrame
def compute_dhr(brf_norm, angles):
    dhr = []
    for i in range(brf_norm.shape[1]):  # Loop over wavelengths
        brf_values = []
        for angle in angles:
            index = olivine_63_30[olivine_63_30[view_angle_column] == angle].index[0]
            brf_value = brf_norm.iloc[index, i]
            brf_values.append(brf_value * cos_theta[angles == angle])
        dhr.append(sum(brf_values) * delta_theta)
    return np.array(dhr)

# Calculate DHR for all configurations
dhr_63_30 = compute_dhr(brf_63_30_norm, selected_angles)
dhr_63_60 = compute_dhr(brf_63_60_norm, selected_angles)
dhr_600_30 = compute_dhr(brf_600_30_norm, selected_angles)
dhr_600_60 = compute_dhr(brf_600_60_norm, selected_angles)

# Plot DHR for each configuration
def plot_dhr(wavelengths, dhr, title):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, dhr, label=title)
    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("DHR")
    plt.grid()
    plt.legend()
    plt.show()

# Generate DHR plots
plot_dhr(wavelengths, dhr_63_30, "DHR for 63–300 µm (θi = 30°)")
plot_dhr(wavelengths, dhr_63_60, "DHR for 63–300 µm (θi = 60°)")
plot_dhr(wavelengths, dhr_600_30, "DHR for 600–1000 µm (θi = 30°)")
plot_dhr(wavelengths, dhr_600_60, "DHR for 600–1000 µm (θi = 60°)")


# In[ ]:




