# Wine Classification Project

Introduction to Artificial Intelligence  
Warsaw University of Technology, Summer 2025  
Lab 4: Regression and Classification

## Authors

- Bora ILCI
- Kaan Emre KARA

## Project Description

This project implements a machine learning classifier for the wine dataset. It uses Support Vector Machine (SVM) and Random Forest algorithms to classify wines into types based on their chemical properties.

The project includes:

- Data preprocessing and feature analysis
- Visualization of feature distributions and correlations
- Model training with parameter tuning using 4-fold cross-validation
- Model evaluation and comparison
- Generating visual reports of the results

## Setup and Installation

1. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   ```

2. Activate the virtual environment:

   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:

   ```
   python main.py
   ```

2. The script will perform:

   - Data loading and analysis
   - Feature visualization
   - Cross-validated model training for both SVM and Random Forest
   - Performance evaluation and comparison
   - Generation of visualization plots

3. Outputs:
   - Console output with performance metrics and parameter tuning results
   - Individual PNG files for each visualization

## Visualization Gallery

To view an interactive gallery of all plots, uncomment the last line in the `main.py` file:

```python
# Display visualization gallery (uncomment to use)
plot_manager.show_gallery()
```

## Files Description

- `main.py`: The main script that executes the entire pipeline
- `visualization.py`: Contains the `PlotManager` class for creating and managing visualizations
- `requirements.txt`: Lists all required Python packages
- `wine_analysis_report.pdf`: Output file containing all visualizations
