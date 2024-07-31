# Diamond-Price-Prediction


## Introduction

Diamond merchants often face challenges in accurately determining the price of diamonds due to the numerous features that need to be considered. This project aims to develop the most efficient model for predicting diamond prices based on various attributes such as carat, cut, color, clarity, and other measurements.

## Problem Summary

Diamonds are rare and valuable, and their pricing is complex. The value of a diamond is typically assessed by a gemologist, who evaluates various features. Our goal is to create a model that can predict diamond prices with high accuracy, using a dataset of approximately 54,000 round-cut diamonds with 10 features. The features include carat, cut, color, clarity, depth, table, price, and measurements (x, y, z).


## Dataset

  * Dataset Name: Diamonds Prices
  
  * Source: [Kaggle Diamonds Prices Dataset](https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices

  * Description: The dataset contains 53,940 diamonds with 10 features: carat, cut, color, clarity, depth, table, price, x (length in mm), y (width in mm), and z (depth in mm).


## Preparation & Preprocessing

  ### Categorical Data
  
  * Decoding categorical variables such as cut, color, and clarity.

  ### Numerical Data
  
  * Checking for NULL values and cleaning redundant data.

  * Scaling the data to minimize outliers.

  * Visualizing data to identify correlations with price.
    
  ### Libraries Used
  
  * numpy

  * pandas
    
  * seaborn


## Methods
I employed three machine learning models to predict diamond prices:

### Linear Regression

  * Accuracy: 92.344%
    
  * Mean Squared Error: 0.278554
    
### Decision Tree Regressor
  
  * Accuracy: 96.536%
    
  * Mean Squared Error: 0.187384
    
### Random Forest Regressor
  
  * Accuracy: 97.909%
  
  * Mean Squared Error: 0.145573


## Results

The Random Forest Regressor outperformed the other models with the highest accuracy and lowest mean squared error.


## Conclusion

The Random Forest Regressor proved to be the most effective model for predicting diamond prices, achieving the highest accuracy among the tested models.


## References

- [Kaggle Diamonds Prices Dataset](https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices)


## Notebooks Description

  ### Image_prediction_CNN.ipynb

  This notebook contains the implementation of Convolutional Neural Networks (CNNs) for predicting the types of expensive gemstones from images. It includes data preprocessing, model architecture, training,     and evaluation.

  ### Edit_Image_folders.ipynb

  This notebook is used for organizing and managing the images of diamonds into their respective folders based on types (round, cushion, emerald, heart, oval, radiant). It includes code for renaming, sorting,  and categorizing images.

  ### diamond.py

  This notebook focuses on exploratory data analysis (EDA) and initial preprocessing of the diamond dataset. It includes visualizations to understand the distribution and relationships of different features with the diamond price.

  ### Diamond_price_predection.ipynb

  This notebook contains the implementation of the three machine learning models (Linear Regression, Decision Tree Regressor, and Random Forest Regressor) for predicting diamond prices. It includes data preprocessing, model training, evaluation, and comparison of results.


## Repository Structure

    Diamond-Price-Prediction/
    ├── all_images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── diamonds/
    │    ├── images/
    │    │   ├── cushion/
    │    │   │   ├── image1.jpg
    │    │   │   ├── image2.jpg
    │    │   │   └── ...
    │    │   ├── emerald/
    │    │   │   ├── image1.jpg
    │    │   │   ├── image2.jpg
    │    │   │   └── ...
    │    │   ├── heart/
    │    │   │   ├── image1.jpg
    │    │   │   ├── image2.jpg
    │    │   │   └── ...
    │    │   ├── oval/
    │    │   │   ├── image1.jpg
    │    │   │   ├── image2.jpg
    │    │   │   └── ...
    │    │   ├── radiant/
    │    │   │   ├── image1.jpg
    │    │   │   ├── image2.jpg
    │    │   │   └── ...
    │    │   └── round/
    │    │       ├── image1.jpg
    │    │       ├── image2.jpg
    │    │       └── ...
    │    │   ├── data_cushion.csv
    │    │   ├── data_emerald.csv
    │    │   ├── data_heart.csv
    │    │   ├── data_oval.csv
    │    │   ├── data_radiant.csv
    │    │   └── data_round.csv
    ├── notebooks/
    │   ├── Image_prediction_CNN.ipynb
    │   ├── Edit_Image_folders.ipynb
    │   ├── diamond.ipynb
    │   ├── Diamond_price_predection.ipynb



