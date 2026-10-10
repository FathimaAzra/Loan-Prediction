
from flask import Flask, render_template, request
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# 1. Create sample dataset
random.seed(42)
data = []

for i in range(1000):
    income = random.randint(15000, 150000)
    co_income = random.randint(0, 100000)
    loan = random.randint(100000, 5000000)
    term = random.choice([12, 24, 36, 48, 60])
    credit = random.choice([0, 1])
    employed = random.choice([0, 1])

    total_income = income + co_income
    payment_ratio = (loan / term) / total_income

    # Create sample loan status
    if credit == 1 and payment_ratio < 0.40 and total_income >= 50000:
        status = 1
    else:
        status = 0

    data.append([
        income, co_income, loan, term, credit, employed, status
    ])

# 2. Convert data into a DataFrame
df = pd.DataFrame(data, columns=[
    "Income", "CoIncome", "Loan", "Term",
    "Credit", "Employed", "Status"
])

# 3. Separate input and output
X = df.drop("Status", axis=1)
y = df["Status"]

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model trained!")
print("Test accuracy:", round(accuracy * 100, 2), "%")


# 6. Connect Flask with HTML
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    rejection_reasons = []

    if request.method == "POST":
        try:
            income = float(request.form["applicant_income"])
            co_income = float(request.form["coapplicant_income"])
            loan = float(request.form["loan_amount"])
            term = int(request.form["loan_term"])
            credit = int(request.form["credit_history"])
            employed = int(request.form["self_employed"])

            # Validate inputs
            if income <= 0 or co_income < 0 or loan <= 0 or term <= 0:
                raise ValueError

            if credit not in [0, 1] or employed not in [0, 1]:
                raise ValueError

            # Prepare input for the model
            input_data = pd.DataFrame([[
                income, co_income, loan, term, credit, employed
            ]], columns=[
                "Income", "CoIncome", "Loan", "Term", "Credit", "Employed"
            ])

            # Predict
            result = model.predict(input_data)[0]

            total_income = income + co_income
            payment_ratio = (loan / term) / total_income

            if result == 1:
                prediction = "Likely Approved"
            else:
                prediction = "Likely Rejected"

                if credit == 0:
                    rejection_reasons.append(
                        "Credit history is poor or unavailable."
                    )

                if payment_ratio >= 0.40:
                    rejection_reasons.append(
                        "Loan payment is high compared with income."
                    )

                if total_income < 50000:
                    rejection_reasons.append(
                        "Combined income is below Rs. 50,000."
                    )

                if not rejection_reasons:
                    rejection_reasons.append(
                        "The combination of input details led to rejection."
                    )

        except (ValueError, TypeError, KeyError):
            error = "Please enter valid values in all fields."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        rejection_reasons=rejection_reasons,
        accuracy=round(accuracy * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
