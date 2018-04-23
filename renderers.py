from flask import render_template


def render_predict_form():
    classifiers = [
        {"name": "Support Vector Machine", "ref": "svm"},
        {"name": "Random Forest", "ref": "rf"},
        {"name": "Gaussian Process", "ref": "gp"},
        {"name": "Neural Network", "ref": "nn"},
    ]

    features = [
        [13.28, 1.64, 2.84, 15.5, 110.0, 2.60, 2.68, 0.34, 1.36, 4.60, 1.09, 2.78, 880.0],
        [12.51, 1.73, 1.98, 20.5, 85.0, 2.20, 1.92, 0.32, 1.48, 2.94, 1.04, 3.57, 672.0],
        [13.49, 3.59, 2.19, 19.5, 88.0, 1.62, 0.48, 0.58, 0.88, 5.7, 0.81, 1.82, 580.0],
    ]
    return render_template('predict_form.html', features=features, classifiers=classifiers)

def render_prediction(prediction):
    return render_template("predict_show.html", prediction=prediction)
