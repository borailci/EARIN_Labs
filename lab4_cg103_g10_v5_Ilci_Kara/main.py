import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import f_classif

# Import visualization tools
from visualization import PlotManager


def main():
    # Part 1: Data Loading and Exploration
    print("Loading and exploring the wine dataset...")
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names

    # Create DataFrame for easier data manipulation
    wine_df = pd.DataFrame(X, columns=feature_names)
    wine_df["target"] = y

    # Print basic dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Features: {feature_names}")

    # Initialize the plot manager
    plot_manager = PlotManager(save_individual_plots=True)

    # Part 2: Data Preparation and Analysis
    print("\nAnalyzing feature distributions and correlations...")

    # Visualize feature distributions
    plot_manager.plot_feature_distributions(wine_df.iloc[:, :-1])

    # Plot correlation matrix between features
    plot_manager.plot_correlation_matrix(wine_df.iloc[:, :-1])

    # Check for feature correlation and decide if we need to remove any features
    correlation_matrix = wine_df.iloc[:, :-1].corr()
    high_correlation = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    high_corr_pairs = high_correlation.stack()[abs(high_correlation.stack()) > 0.8]

    if not high_corr_pairs.empty:
        print("\nHighly correlated feature pairs (>0.8):")
        for idx, value in high_corr_pairs.items():
            print(f"{idx[0]} and {idx[1]}: {value:.2f}")
        print(
            "Note: We will keep all features for now as the dataset is small and our models can handle correlation."
        )
    else:
        print("\nNo features with correlation > 0.8 found.")

    # Part 3: Data Splitting
    print("\nSplitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Part 4: Feature Scaling
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature importance analysis using ANOVA F-test
    print("\nCalculating feature importance using ANOVA F-test...")
    f_values, p_values = f_classif(X_train, y_train)

    # Plot feature importance
    importance_df = plot_manager.plot_feature_importance(feature_names, f_values)
    print("Top 5 most important features based on F-test:")
    print(importance_df.head(5))

    # Part 5: Model Selection and Training - Support Vector Machine
    print("\n===== Training Support Vector Machine Model =====")

    # Parameter tuning for SVM using cross-validation
    print("Performing grid search for SVM parameters with 4-fold cross-validation...")
    svm_param_grid = {
        "C": [1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    }

    svm_grid = GridSearchCV(
        SVC(),
        svm_param_grid,
        cv=4,  # 4-fold cross-validation as required
        scoring="accuracy",
        verbose=1,
        return_train_score=True,
    )

    svm_grid.fit(X_train_scaled, y_train)

    # Display SVM cross-validation results
    print(f"\nBest SVM parameters: {svm_grid.best_params_}")
    print(f"Best SVM cross-validation accuracy: {svm_grid.best_score_:.4f}")

    # Create a table of SVM results for different parameter combinations
    print("\nSVM Parameter Tuning Results:")
    svm_results = pd.DataFrame(svm_grid.cv_results_)
    svm_results_table = svm_results[["params", "mean_test_score", "std_test_score"]]
    svm_results_table = svm_results_table.sort_values(
        by="mean_test_score", ascending=False
    )
    for i, row in svm_results_table.head(7).iterrows():
        params = row["params"]
        mean = row["mean_test_score"]
        std = row["std_test_score"]
        print(f"Parameters: {params}, Mean CV Accuracy: {mean:.4f}, Std: {std:.4f}")

    # Get the best SVM model
    svm_best = svm_grid.best_estimator_

    # Part 6: Model Selection and Training - Random Forest
    print("\n===== Training Random Forest Model =====")

    # Parameter tuning for Random Forest using cross-validation
    print(
        "Performing grid search for Random Forest parameters with 4-fold cross-validation..."
    )
    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5],
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=4,  # 4-fold cross-validation as required
        scoring="accuracy",
        verbose=1,
        return_train_score=True,
    )

    rf_grid.fit(X_train_scaled, y_train)

    # Display Random Forest cross-validation results
    print(f"\nBest Random Forest parameters: {rf_grid.best_params_}")
    print(f"Best Random Forest cross-validation accuracy: {rf_grid.best_score_:.4f}")

    # Create a table of Random Forest results for different parameter combinations
    print("\nRandom Forest Parameter Tuning Results:")
    rf_results = pd.DataFrame(rf_grid.cv_results_)
    rf_results_table = rf_results[["params", "mean_test_score", "std_test_score"]]
    rf_results_table = rf_results_table.sort_values(
        by="mean_test_score", ascending=False
    )
    for i, row in rf_results_table.head(7).iterrows():
        params = row["params"]
        mean = row["mean_test_score"]
        std = row["std_test_score"]
        print(f"Parameters: {params}, Mean CV Accuracy: {mean:.4f}, Std: {std:.4f}")

    # Get the best Random Forest model
    rf_best = rf_grid.best_estimator_

    # Plot Random Forest feature importance
    plot_manager.plot_feature_importance(
        feature_names,
        rf_best.feature_importances_,
        title="Random Forest Feature Importance",
    )

    # Part 7: Model Evaluation on Test Set
    print("\n===== Model Evaluation on Test Set =====")

    # Generate predictions
    svm_preds = svm_best.predict(X_test_scaled)
    rf_preds = rf_best.predict(X_test_scaled)

    # Compare model performance
    svm_accuracy = accuracy_score(y_test, svm_preds)
    rf_accuracy = accuracy_score(y_test, rf_preds)

    print(f"\nSVM Test Accuracy: {svm_accuracy:.4f}")
    print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")

    # Print classification reports
    print("\nSVM Classification Report:")
    print(classification_report(y_test, svm_preds, target_names=target_names))

    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_preds, target_names=target_names))

    # Plot confusion matrices
    plot_manager.plot_confusion_matrices(y_test, svm_preds, rf_preds)

    # Part 8: Visualization and Report Generation
    print("\n===== Generating Visualizations =====")
    print("\nAnalysis complete!")
    print(
        "You can view an interactive gallery of all plots by uncommenting the last line in main.py."
    )

    # Display visualization gallery (uncomment to use)
    plot_manager.show_gallery()


if __name__ == "__main__":
    main()
