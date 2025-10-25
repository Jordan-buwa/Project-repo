import yaml

# Load YAML config
config_path = "config/config_process.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Print top-level keys
print("Top-level keys in config:", config.keys())

# Verify numerical features
if "numerical_features" in config:
    print("First 5 numerical features:", config["numerical_features"][:5])
else:
    print("Key 'numerical_features' not found!")

# Verify categorical features
if "categorical_features" in config:
    print("First 5 categorical features:", config["categorical_features"][:5])
else:
    print("Key 'categorical_features' not found!")

# Verify combined features
if "combined_features" in config:
    print("Combined features:", config["combined_features"])
else:
    print("Key 'combined_features' not found!")
