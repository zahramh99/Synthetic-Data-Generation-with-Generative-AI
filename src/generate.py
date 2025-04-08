import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from preprocess import preprocess_data

def generate_synthetic_data(generator_path, scaler, num_samples=1000, latent_dim=100):
    """Generate synthetic data using trained generator."""
    generator = load_model(generator_path)
    
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data = generator.predict(noise)
    
    # Rescale to original values
    generated_data_rescaled = scaler.inverse_transform(generated_data)
    
    return pd.DataFrame(generated_data_rescaled, columns=['Usage', 'Notifications', 'Times opened'])

if __name__ == "__main__":
    # Load preprocessor
    _, scaler = preprocess_data('../data/Screentime-App-Details.csv')
    
    # Generate data
    synthetic_data = generate_synthetic_data('generator_model.h5', scaler)
    print(synthetic_data.head())
    
    # Save to CSV
    synthetic_data.to_csv('synthetic_app_usage.csv', index=False)
    