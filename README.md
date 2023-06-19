# Laptop Price Prediction

This project focuses on predicting the price of laptops based on various features such as company, laptop type, RAM size, weight, touchscreen availability, IPS display, screen size, screen resolution, CPU type, and HDD memory size.

## Dataset

The project uses a dataset called `laptop_data.csv`. The dataset contains information about different laptops, including their specifications and corresponding prices. The dataset is read using the Pandas library in Python.

## Data Preprocessing

The dataset undergoes several preprocessing steps to prepare it for analysis and modeling. The steps include:

1. Removing duplicate rows.
2. Dropping rows with missing values.
3. Removing unnecessary columns ('Unnamed: 0').
4. Cleaning and converting the 'Ram' and 'Weight' columns to the appropriate data types.
5. Extracting information from the 'ScreenResolution' column to create new columns 'X_res' and 'Y_res'.
6. Converting the extracted 'X_res' and 'Y_res' columns to integers.
7. Calculating the PPI (Pixels Per Inch) based on the 'X_res', 'Y_res', and 'Inches' columns.

## Exploratory Data Analysis (EDA)

The project performs exploratory data analysis to gain insights into the dataset and the relationships between variables. The EDA includes various visualizations using libraries such as Seaborn and Matplotlib. The visualizations include:

1. Distribution plot of laptop prices.
2. Bar plots of laptop counts by company, laptop type, touchscreen availability, IPS display, CPU brand, RAM size, and operating system.
3. Scatter plots and distribution plots of laptop prices against screen size, weight, and PPI.
4. Heatmap to visualize the correlation between different features.

## Machine Learning Models

The project employs various machine learning models to predict laptop prices based on the given features. The models used include:

1. Linear Regression
2. Ridge Regression
3. K-Nearest Neighbors (KNN)
4. Decision Tree
5. Support Vector Machines (SVM)
6. Random Forest
7. Gradient Boosting

The models are trained using the preprocessed dataset and evaluated using the R2 score and mean absolute error (MEA).

## Usage

To use the trained model for price prediction, follow these steps:

1. Run the code in the Jupyter Notebook or Python environment.
2. After training the models, the program prompts for input:
   - Enter the laptop company.
   - Enter the laptop type.
   - Enter the RAM size in GB (2, 4, 6, 8, 12, 16, 24, 32, or 64).
   - Enter the weight of the laptop in kg.
   - Enter touchscreen availability (Yes or No).
   - Enter IPS display availability (Yes or No).
   - Enter the screen size.
   - Enter the screen resolution (e.g., 1920x1080).
   - Enter the CPU type.
   - Enter the HDD memory size in GB (0, 128, ...).
3. The program will predict the price based on the provided inputs.

Note: The prediction accuracy may vary depending on the trained model and the input data.

## Dependencies

The following dependencies are required to run the code:

- Python 3
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost

You can install the required dependencies using pip or conda package manager.

```shell
pip install numpy pandas seaborn matplotlib scikit-learn xgboost
```

## License

The project is available under the [MIT License](LICENSE).

Please note that this project is for educational purposes and should be used responsibly.
