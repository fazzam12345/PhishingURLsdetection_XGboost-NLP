from src.data_ingestion import load_raw_data
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTrainer


def main():
    # Step 1: Load raw data
    raw_data_path = "../mlops/data/raw/dataset.csv"
    print("Loading raw data...")
    df = load_raw_data(raw_data_path)
    print("Raw data loaded successfully.")

    # Step 2: Perform feature engineering
    print("Performing feature engineering...")
    fe = FeatureEngineering()
    features_df = fe.extract_features(df)
    print("Feature engineering completed.")

    features_file_path = "../mlops/data/processed/features.csv"
    fe.save_features(features_df, features_file_path)
    print("Features saved to CSV file.")

    # Prepare data for model training
    X = features_df.drop("label", axis=1)
    y = features_df["label"]

    # Step 3: Train the model
    print("Training the model...")
    trainer = ModelTrainer(use_smote=True, log_feature_importance=True)
    trainer.train(X, y)
    print("Model training completed.")

    # Step 4: Save the trained model
    model_path = "../mlops/models/model.pkl"
    trainer.save_model(model_path)
    print("Trained model saved to disk.")


if __name__ == "__main__":
    main()
