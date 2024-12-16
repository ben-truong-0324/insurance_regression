# Kaggle Housing ML Project

This project leverages machine learning models to predict housing prices based on the Kaggle House Prices: Advanced Regression Techniques dataset. We explore a variety of techniques, including decision trees, bagging, and boosting, to predict home prices in Ames, Iowa, using the provided explanatory variables. This project covers data preprocessing, model training, evaluation, and result visualization.

## Project Overview

Data is retrieved from https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
In this competition, participants are tasked with predicting the final sale price of homes in Ames, Iowa, based on 79 explanatory variables. These variables cover almost every aspect of residential homes, from architectural features to neighborhood characteristics. The goal is to develop models that can predict home prices accurately using regression techniques.

## Installation and Setup

### **Setting up the Conda Environment**

To set up the project, it is recommended to use the provided `environment.yml` file to create a conda environment with the necessary dependencies.

```bash
conda env create -f environment.yml
conda env update --file environment.yml --prune
conda activate ml_general
python -m rohlik_sales.ml

# download/copy data into ./data/
```

## License

This project is licensed under the MIT License.
MIT License

Copyright (c) 2024 Ben Truong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
