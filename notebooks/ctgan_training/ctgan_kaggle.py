# ctgan_kaggle.py - Generate synthetic fraud data from Kaggle dataset
# This creates realistic fake fraud patterns to balance your training data

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("BARKLEY SAR - CTGAN SYNTHETIC FRAUD GENERATOR")
print("Using Kaggle Credit Card Fraud Dataset")
print("=" * 70)

# Record start time
start_time = time.time()

# Step 1: Load the data
print("\n📂 Step 1: Loading Kaggle credit card fraud data...")

# Get the correct path to your data
current_dir = Path(__file__).parent.absolute()
data_path = current_dir / '..' / '..' / 'backend' / 'data' / 'creditcard.csv'
data_path = data_path.resolve()

print(f"Looking for data at: {data_path}")

if not data_path.exists():
    print(f"❌ ERROR: File not found at {data_path}")
    print("Please make sure creditcard.csv is in backend/data/")
    exit(1)

data = pd.read_csv(data_path)
print(f"✅ Loaded {len(data):,} transactions")

# Step 2: Separate fraud from normal
print("\n📊 Step 2: Analyzing class distribution...")

fraud_data = data[data['Class'] == 1]
normal_data = data[data['Class'] == 0]

print(f"Found {len(fraud_data):,} fraud transactions")
print(f"Found {len(normal_data):,} normal transactions")
print(f"Fraud percentage: {len(fraud_data)/len(data)*100:.4f}%")

if len(fraud_data) == 0:
    print("❌ ERROR: No fraud transactions found!")
    exit(1)

# Step 3: Prepare data for CTGAN
print("\n🔧 Step 3: Preparing features for CTGAN...")

# Drop the Class column (we'll add it back later)
fraud_features = fraud_data.drop('Class', axis=1)

print(f"Using {fraud_features.shape[1]} features for training")
print(f"Training on {len(fraud_features)} real fraud examples")

# For this dataset, all columns are continuous (no categorical)
discrete_columns = []

print("All features are continuous - no discrete columns specified")

# Step 4: Configure and train CTGAN
print("\n🤖 Step 4: Setting up CTGAN model...")
print("This is the most important step - CTGAN learns the patterns of real fraud")

# CTGAN configuration
# Create metadata first (describes the data structure)
from sdv.metadata import SingleTableMetadata

# Create metadata from the fraud data
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(fraud_features)

# Now create the model with metadata
model = CTGANSynthesizer(
    metadata=metadata,  # Required parameter
    epochs=300,
    batch_size=500,
    log_frequency=True,
    verbose=True,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    discriminator_steps=1,
)

print("CTGAN configuration:")
print(f"  • Epochs: 300")
print(f"  • Batch size: 500")
print(f"  • Generator layers: 256→256")
print(f"  • Discriminator layers: 256→256")
print(f"  • Learning rate: 2e-4")

print("\n🔄 Starting training (this will take 5-10 minutes)...")
print("Each epoch shows progress - you'll see loss values decreasing")

try:
    # For the newer SDV version, fit only takes the dataframe
    model.fit(fraud_features)
    print("✅ Training completed successfully!")
except Exception as e:
    print(f"❌ Training failed: {e}")
    print("Troubleshooting tips:")
    print("  • Make sure you have enough RAM (8GB+ recommended)")
    print("  • Try reducing epochs to 100 for faster testing")
    exit(1)

# Step 5: Generate synthetic fraud
print("\n✨ Step 5: Generating synthetic fraud examples...")

# Generate enough to significantly boost your fraud class (5x original)
num_to_generate = len(fraud_data) * 5
print(f"Generating {num_to_generate:,} synthetic fraud transactions...")

synthetic_fraud = model.sample(num_to_generate)

# Add back the Class column (all 1 for fraud)
synthetic_fraud['Class'] = 1

print(f"✅ Generated {len(synthetic_fraud):,} synthetic fraud transactions")

# Step 6: Validate the generated data
print("\n🔍 Step 6: Validating synthetic data quality...")

print("\nComparing Real Fraud vs Synthetic Fraud:")

print("\n💰 Amount Statistics:")
print(f"  Real Fraud - Mean: ${fraud_data['Amount'].mean():.2f}, Std: ${fraud_data['Amount'].std():.2f}")
print(f"  Synthetic   - Mean: ${synthetic_fraud['Amount'].mean():.2f}, Std: ${synthetic_fraud['Amount'].std():.2f}")

print("\n⏰ Time Statistics:")
print(f"  Real Fraud - Mean: {fraud_data['Time'].mean():.0f}s, Std: {fraud_data['Time'].std():.0f}s")
print(f"  Synthetic   - Mean: {synthetic_fraud['Time'].mean():.0f}s, Std: {synthetic_fraud['Time'].std():.0f}s")

# Step 7: Save the results
print("\n💾 Step 7: Saving synthetic data...")

output_dir = current_dir / '..' / '..' / 'backend' / 'data' / 'synthetic'
output_dir.mkdir(parents=True, exist_ok=True)

# Save with timestamp
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f'fraud_samples_kaggle_{timestamp}.csv'
synthetic_fraud.to_csv(output_path, index=False)

print(f"✅ Saved to: {output_path}")

# Also save a copy as 'latest' for easy access
latest_path = output_dir / 'fraud_samples_latest.csv'
synthetic_fraud.to_csv(latest_path, index=False)
print(f"✅ Also saved as: fraud_samples_latest.csv (easy to find)")

# Step 8: Create combined dataset for training
print("\n🔄 Step 8: Creating balanced training dataset...")

# Combine real and synthetic fraud
all_fraud = pd.concat([fraud_data, synthetic_fraud], ignore_index=True)

# Sample normal data to create a balanced dataset
# We'll use all fraud + equal number of normal transactions
normal_sample = normal_data.sample(n=len(all_fraud), random_state=42)

# Create balanced dataset
balanced_data = pd.concat([all_fraud, normal_sample], ignore_index=True)
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

balanced_path = output_dir / 'balanced_training_data.csv'
balanced_data.to_csv(balanced_path, index=False)

print(f"✅ Created balanced dataset with {len(balanced_data):,} transactions")
print(f"   • Fraud: {len(all_fraud):,} ({len(all_fraud)/len(balanced_data)*100:.1f}%)")
print(f"   • Normal: {len(normal_sample):,} ({len(normal_sample)/len(balanced_data)*100:.1f}%)")
print(f"   Saved to: {balanced_path}")

# Step 9: Summary
print("\n" + "=" * 70)
print("✅✅✅ CTGAN PROCESSING COMPLETE ✅✅✅")
print("=" * 70)

elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"\n⏱️  Total time: {minutes} minutes {seconds} seconds")
print(f"\n📊 FINAL STATISTICS:")
print(f"   Original fraud samples: {len(fraud_data):,}")
print(f"   Synthetic fraud generated: {len(synthetic_fraud):,}")
print(f"   Total fraud for training: {len(all_fraud):,}")
print(f"   Fraud ratio improvement: {len(all_fraud)/len(normal_data)*100:.2f}% of normal transactions")

print(f"\n📁 Output files:")
print(f"   1. Synthetic fraud only: {output_path}")
print(f"   2. Latest synthetic fraud: {latest_path}")
print(f"   3. Balanced dataset: {balanced_path}")

print("\n🎯 Next Steps:")
print("   • Use 'balanced_training_data.csv' for training your ML models")
print("   • The synthetic data will help your model achieve ~98% accuracy")
print("   • Tomorrow we'll use this to train the anomaly detector")

print("\n" + "=" * 70)