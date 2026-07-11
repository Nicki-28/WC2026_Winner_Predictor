# 🏆 FIFA World Cup 2026 - Predictive ML Model

## Overview
A Machine Learning project designed to predict the outcomes of the 2026 FIFA World Cup. Built entirely from scratch as a personal deep-dive into Data Science, this project processes historical match data from 1930 onwards to build progressive, dynamic features (such as historical goal averages). By simulating a "time machine" approach during feature engineering, the model strictly avoids Data Leakage, proving that with the right logic, raw football history can be turned into a smart predictive algorithm.

## The Learning Journey & Core Concepts
Throughout the development of this model, I focused on understanding the "why" behind the code, tackling complex data challenges such as:

* **Feature Engineering:** Transforming raw match results into dynamic, progressive statistics (like historical goal averages prior to kick-off).
* **Preventing Data Leakage:** Implementing a strict "time machine" logic loop to ensure the model only learns from the past relative to each match, effectively avoiding the most common trap for Data Science beginners.
* **Mathematical Correlation:** Generating correlation matrices to validate intuitive football knowledge with hard statistical evidence.
* **Train/Test Split:** Protecting the model's learning process by hiding modern data (2018-2022) to ensure it genuinely predicts the future rather than memorizing historical scores.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn
* **Environment:** Jupyter Notebook (Data Exploration & Processing) & Python Scripts (Model Training & Simulation)

## Data Strategy & Model Architecture
The model trains on historical international football matches from 1930 to 2022. It filters out statistical "noise" (like IDs, referees, or stadium capacities) and engineers smart features to predict the target variable of future matches.

* **Algorithm:** Random Forest Classifier (an ensemble of decision trees to capture the non-linear chaos of football).
* **Performance:** Achieved ~60% accuracy on predicting exact match outcomes (Win/Loss/Draw) against unseen test data, significantly outperforming random chance (33%) and competing with professional betting algorithms.

## 🔮 Tournament Simulation & Results
A custom "Tournament Cascade" script was built to simulate the 2026 knockout brackets. The script intelligently translates winners from one round to the next (Round of 32 -> Final) using the trained algorithm and updated team statistics.

**2026 Prediction Highlights:**
* **Champion:** Spain 🇪🇸
* **Surprise Finalist:** Norway 🇳🇴 (A statistical "dark horse" driven by extreme recent goal-scoring metrics).
* **Third Place:** France 🇫🇷
