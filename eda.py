import os 
import numpy as np
import pandas as pd

df= pd.read_csv("C:/Users/catji/Downloads/archive/weather_prediction_dataset.csv")
print(df.head())

df_transformed = df.copy()
numerical_cols = df.select_dtypes(include=['number'])
skewness_values = numerical_cols.skew()

skew_threshold = 0.75
highly_skewed_features = skewness_values[abs(skewness_values) > skew_threshold]
# Filter out features containing 'Pressure' or 'Extreme precipitation'
# Corrected filtering logic to be case-insensitive for 'pressure' and 'precipitation'
features_to_transform = [col for col in highly_skewed_features.index
                         if 'pressure' not in col.lower() and 'precipitation' not in col.lower()]

print(f"Applying transformations to {len(features_to_transform)} highly skewed features (excluding pressure and precipitation columns)...")

for feature in features_to_transform:
    original_skew = highly_skewed_features[feature]

    # Apply transformations based on skewness direction
    if original_skew > 0:  # Positively skewed
        # Use log1p as it handles zeros safely
        df_transformed[feature] = np.log1p(df_transformed[feature])
    else:  # Negatively skewed
        # Apply reflective transformation: max_val + 1 - x, then log1p
        max_val = df_transformed[feature].max()
        df_transformed[feature] = np.log1p(max_val + 1 - df_transformed[feature])

print("Transformations applied. Recalculating skewness for transformed features...")

# Recalculate skewness for the transformed features
new_skewness = df_transformed[features_to_transform].skew()

# Compare original and new skewness
skewness_comparison_filtered = pd.DataFrame({
    'Original Skewness': highly_skewed_features[features_to_transform],
    'Transformed Skewness': new_skewness
})

print("\nSkewness before and after transformation for filtered highly skewed features:")
print(skewness_comparison_filtered)
