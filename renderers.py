from flask import render_template
import models as dbHandler
from computation import build_model

classifiers = [
    {"name": "Support Vector Machine", "ref": "svm"},
    {"name": "Random Forest", "ref": "rf"},
    {"name": "Gaussian Process", "ref": "gp"},
    {"name": "Neural Network", "ref": "nn"},
]

def render_predict_form():
    features = [
        [13.28, 1.64, 2.84, 15.5, 110.0, 2.60, 2.68, 0.34, 1.36, 4.60, 1.09, 2.78, 880.0],
        [12.51, 1.73, 1.98, 20.5, 85.0, 2.20, 1.92, 0.32, 1.48, 2.94, 1.04, 3.57, 672.0],
        [13.49, 3.59, 2.19, 19.5, 88.0, 1.62, 0.48, 0.58, 0.88, 5.7, 0.81, 1.82, 580.0],
    ]
    return render_template('predict_form.html', features=features, classifiers=classifiers)

def render_prediction(prediction):
    return render_template("predict_show.html", prediction=prediction)

def render_create_model_form(errors, csv_file=None, classifier_type=None, file_name=None, num_features=None):
    return render_template("models_create_form.html",
                           classifiers=classifiers,
                           errors=errors,
                           csv_file=csv_file,
                           classifier_type=classifier_type,
                           file_name=file_name,
                           num_features=num_features)

def render_create_model(form, files):
    errors = []

    # csv_file = form.get("csv_file")
    csv_file = files['csv_file']
    print str(csv_file)
    classifier_type = form.get("classifier_type")
    file_name = form.get("file_name")
    num_features = form.get("num_features")

    if all(p is not None for p in [csv_file, classifier_type, file_name, num_features]):
        id, errors = build_model(csv_file, classifier_type, file_name, num_features)
        # train and save model
        # if no error: error = False

    if len(errors) > 0:
        return render_create_model_form(errors, csv_file, classifier_type, file_name, num_features)
        # redirect to GET /models/new pass in error
    else:
        return render_model_show(id)
        # render GET /models/123456

def render_model_show(id):
    model = dbHandler.retrieveModel(id)
    print model
    classifier_type = ""
    file_name = ""
    num_features = "13"
    features = [
        [13.28, 1.64, 2.84, 15.5, 110.0, 2.60, 2.68, 0.34, 1.36, 4.60, 1.09, 2.78, 880.0],
        [12.51, 1.73, 1.98, 20.5, 85.0, 2.20, 1.92, 0.32, 1.48, 2.94, 1.04, 3.57, 672.0],
        [13.49, 3.59, 2.19, 19.5, 88.0, 1.62, 0.48, 0.58, 0.88, 5.7, 0.81, 1.82, 580.0],
    ]
    return render_template("models_show.html", id=id, classifier_type=classifier_type, file_name=file_name, num_features=num_features, features=features)

def render_index_model():
    models = dbHandler.retrieveModels()
    return render_template("models_index.html", models=models)
