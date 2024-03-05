# DVA-041 Project
# Stock Analysis and Forecasting

## Setup
Make sure you have some version of conda and Python 3.11 installed. Then, run
the following commands to create a new environment and install the required
dependencies and activate the environment:
```bash
conda env create -f environment.yml
conda activate dva-041
```
In turn, if you need to update dependencies, you can run:
```bash
conda env export -f environment.yml
```

## Usage
If you have not already done so, you will need to download the relevant data:
```bash
python3 download_data.py
```