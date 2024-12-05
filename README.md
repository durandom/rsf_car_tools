# RSF Force Feedback Settings Manager

A Python tool for managing and predicting Force Feedback (FFB) settings in RallySimFans (RSF).

## Overview

This tool helps manage and optimize Force Feedback settings for cars in RallySimFans by:

- Analyzing existing FFB configurations across your car collection
- Training machine learning models to predict optimal FFB settings
- Clustering cars based on physical characteristics
- Visualizing car statistics and distributions
- Managing undriven cars

## Features

- **FFB Analysis**: Analyzes custom Force Feedback settings for tarmac, gravel, and snow surfaces
- **Machine Learning**: Uses polynomial regression to predict FFB settings based on car characteristics
- **Clustering**: Groups similar cars together for better analysis
- **Statistics**: Generates visualizations for:
  - Car weight distribution
  - Drivetrain types
  - Steering wheel angles
- **Sample Selection**: Intelligently selects representative car samples for FFB training
- **Validation**: Tests prediction accuracy against known FFB settings

## Requirements

- Python 3.7+
- Required packages:
  - numpy
  - scikit-learn
  - plotext
  - rich
  - loguru
  - configobj

## Installation

```bash
pipenv install
```

## Usage

Basic usage:

```bash
python power_poly.py /path/to/rsf/installation [options]
```

Options:
- `--verbose, -v`: Increase verbosity (can be used multiple times)
- `--stats`: Plot statistics (comma-separated: weight,drivetrain,steering)
- `--train`: Train FFB prediction models
- `--validate`: Validate FFB predictions
- `--undriven`: List undriven cars
- `--select-sample [N]`: Select N cars from each cluster (default: 3)
- `--html FILE`: Save console output to HTML file

Examples:

```bash
# Show car statistics
python power_poly.py "C:/Games/RSF" --stats weight,drivetrain

# Train and validate FFB models
python power_poly.py "C:/Games/RSF" --train --validate

# Select training sample and save report
python power_poly.py "C:/Games/RSF" --select-sample 5 --html report.html
```

## How It Works

1. **Data Collection**:
   - Reads car configurations from RSF's personal.ini
   - Loads technical data from cars.json and cars_data.json
   - Extracts key features: weight, steering angle, drivetrain type

2. **Analysis**:
   - Groups cars into clusters based on physical characteristics
   - Identifies cars with custom FFB settings
   - Generates statistical visualizations

3. **Prediction**:
   - Trains separate models for tarmac, gravel, and snow surfaces
   - Uses polynomial regression with feature scaling
   - Validates predictions against known settings

4. **Output**:
   - Console tables with detailed statistics
   - Optional HTML reports
   - Plots of car distributions
