import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import os

input_path = "real_sample_data.csv"
output_path = "synthetic_data.csv"
num_rows = 1000

def generate_synthetic_data(input_path, output_path, num_rows=1000):
    print(f"Loading dataset from: {input_path}")
    real_data = pd.read_csv(input_path, encoding="utf-8-sig")
    metadata = Metadata.detect_from_dataframe(data=real_data)

    print("Fitting SDV GaussianCopulaSynthesizer model...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)

    print(f"Generating {num_rows} synthetic rows...")
    synthetic_data = synthesizer.sample(num_rows)

    common_columns = [col for col in [
        'Highwall',
        'Height [m]',
        'Length [m]',
        'Average slope angle [°]',
        'Average slope roughness [°]',
        'Number of virtual trajectories'
    ] if col in synthetic_data.columns]

    synthetic_data = synthetic_data[common_columns]

    synthetic_data = calculate_risk(synthetic_data)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data saved to: {output_path}")


def calculate_risk(data: pd.DataFrame) -> pd.DataFrame:
    if 'Height [m]' in data.columns:
        data['H_norm'] = (data['Height [m]'] - data['Height [m]'].min()) / (data['Height [m]'].max() - data['Height [m]'].min())
    if 'Length [m]' in data.columns:
        data['L_norm'] = (data['Length [m]'] - data['Length [m]'].min()) / (data['Length [m]'].max() - data['Length [m]'].min())
    if 'Average slope angle [°]' in data.columns:
        data['Theta_norm'] = (data['Average slope angle [°]'] - data['Average slope angle [°]'].min()) / (data['Average slope angle [°]'].max() - data['Average slope angle [°]'].min())
    if 'Average slope roughness [°]' in data.columns:
        data['R_norm'] = 1 - ((data['Average slope roughness [°]'] - data['Average slope roughness [°]'].min()) / (data['Average slope roughness [°]'].max() - data['Average slope roughness [°]'].min()))
    if 'Number of virtual trajectories' in data.columns:
        data['T_norm'] = (data['Number of virtual trajectories'] - data['Number of virtual trajectories'].min()) / (data['Number of virtual trajectories'].max() - data['Number of virtual trajectories'].min())

    norm_cols = ['H_norm', 'L_norm', 'Theta_norm', 'R_norm', 'T_norm']
    for col in norm_cols:
        if col not in data.columns:
            data[col] = 0

    data['Risk_Score'] = (
        0.3 * data['H_norm'] +
        0.2 * data['L_norm'] +
        0.3 * data['Theta_norm'] +
        0.2 * data['R_norm'] +
        0.2 * data['T_norm']
    )

    def categorize(score):
        if score < 0.33:
            return "Low"
        elif score < 0.66:
            return "Medium"
        else:
            return "High"

    data['Risk_Level'] = data['Risk_Score'].apply(categorize)

    return data

if __name__ == "__main__":
    generate_synthetic_data(input_path, output_path, num_rows)
