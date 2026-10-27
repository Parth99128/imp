MACHINE LEARNING PRACTICALS — 14 EASY PYTHON CODES
==================================================

This folder contains 14 separate Python scripts covering both versions.
Practical 1, 2, 3 and 9 are shared; Practical 4–8 have Version A and Version B.

HOW TO RUN
----------
1. Install Python 3, or use Google Colab.
2. Install common libraries in a terminal:
   pip install pandas numpy matplotlib scikit-learn mlxtend
3. Put the required CSV dataset in the same folder as the script.
4. Run one script at a time:
   python 02_linear_regression_age_price.py

In Google Colab, upload the relevant CSV file and paste/run the script.
If a library is missing in Colab, run:
   !pip -q install pandas numpy matplotlib scikit-learn mlxtend

DATASET FILENAMES EXPECTED
--------------------------
Student preprocessing: Student.csv
Iris: Iris.csv
Loan approval: loan__approval_data.csv
Used car: used_car_dataset.csv
Student performance: StudentsPerformance.csv
Groceries: Groceries_dataset.csv
Mall customers: Mall_Customers.csv

IMPORTANT DATA NOTES
--------------------
- Your Iris file columns should include:
  SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species.
- Student Performance columns should include:
  math score, reading score, writing score.
- Loan Approval target column should be Loan_Status.
- Used-car columns should include:
  car_price_in_rupees, kms_driven, fuel_type, city, year_of_manufacture.
- Groceries columns should include:
  Member_number, Date, itemDescription.
- The Student and Mall Customers data were supplied as PDFs. Convert their tables
  to CSV before running their scripts. The code cannot read a PDF table as a CSV.
- Version 2 Practical 4 uses EXAMPLE values because the exact arrays in the sheet
  need to be copied in. Replace the arrays in 04B_logistic_regression_student.py
  with the exact Study_Hours, Attendance and Pass arrays from your practical sheet.
- Rename your dataset files to the expected names above, or change the filename
  inside the code to your actual filename.

WHAT EACH FILE DOES
-------------------
01: Missing-value handling, encoding and scaling.
02: Linear regression for the given Age/Price values.
03: Gradient descent for y=(x+3)^2.
04A: Logistic regression on Iris.
04B: Logistic regression on Student study/attendance data.
05A: Decision tree on Iris.
05B: Decision tree on Loan Approval.
06A: KNN regression on used-car prices.
06B: KNN classification on Iris.
07A: K-Means on Student Performance with K=5.
07B: K-Means on Mall Customers with K=5.
08A: Association rules from five given transactions.
08B: Association rules from Groceries dataset.
09: Elbow method on Student Performance.

TIP FOR THE PRACTICAL EXAM
--------------------------
First learn the aim, input columns, main model line, and output metrics.
For KNN, K-Means and association rules, know what k/support/confidence/lift mean.
