# Predict Candidate Fit Score

## Steps to Run
1. Install dependencies:
   ```
   pip install pandas scikit-learn joblib
   ```

2. Train model:
   ```
   python main.py
   ```

3. Use prediction function in Python:
   ```python
   from main import predict
   print(predict("candidate text", "job text"))
   ```
