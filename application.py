from flask import Flask, request, render_template
from computation import svm_test, make_prediction
from renderers import render_predict_form, render_prediction, render_create_model, render_create_model_form, render_model_show, render_index_model
import models as dbHandler

app = Flask(__name__)
# $ source restful_data_science/bin/activate
# $ FLASK_APP=application.py flask run

# APP FLOW
#
# Land on models list GET / GET /models
# Can created one GET /models/new -> POST /models
# Can delete one DELETE /models/123456 (either form list view or from single view)
# Can click on one GET /models/123456
# Can make prediction GET /models/123456/predict -> POST /results
# Can view that result GET /models/123456/results/789456
# ? Can view all results? GET /models/123456/results
#
# So we need a db with two models:
# Models:
#     classifier type
#     params
#     featureCount
#     filename
#
# Results:
#     modelId
#     features
#     outputLabel
#
#
# GET /models/new
# select classifier
# classifier parameters
# uploads csv file
# parameters necessary to parse csv file
#
# save -> POST /models
# build model, pass in train/test ratio, output accuracy
# store model on local disk, using pickle
# redirect -> GET /models/123456
#
# GET /models/123456
# generated form to make prediction
# form should have dynamically generated fields based on the training data
# submit -> POST /models/123456/predictions
# load model from pickle
# make prediction, save prediction
# redirect to GET /models/123456/prediction/789456

# TODO: make home page
@app.route("/")
@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)


@app.route("/models", methods=["GET", "POST"])
def models():
    if request.method == "POST":
        return render_create_model(request.form, request.files)
        # build model based on params, and train-test data
        # save model with pickle
        # save model record on db
        # redirect to GET /models or GET /models/123456
    else:
        # return str(dbHandler.retrieveModels())
        return render_index_model()
        # get different models from db
        # render model list view

@app.route("/models/new", methods=["GET"])
def models_create():
    return render_create_model_form([])

@app.route("/models/<int:model_id>", methods=["GET", "POST"])
def models_predict(model_id):
    if request.method == "POST":
        # call computation function
        # DONE - clean features (move to computation file?)
        # load trained model based on model id
        # predict wine
        # redirect to prediction page
        sorted_feature_name = sorted(request.form, key=lambda k: int(k.replace('feature', '')))
        sorted_features_values = [request.form[f] for f in sorted_feature_name]
        return "sorted features: " + str(sorted_features_values)
    else:
        return render_model_show(model_id)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        classifier_type = request.form.get("classifier_type")
        features = []

        for i in range(13):
            key = "feature" + str(i)
            features.append(request.form.get(key))

        prediction = make_prediction(classifier_type, features)
        return render_prediction(prediction)
    else:
        return render_predict_form()


@app.route("/testSQLite")
def testSQLite():
    # dbHandler.insertModel("rf", "13", "first_rf_test.txt")
    res = dbHandler.retrieveModels()
    return str(res)


@app.route("/svm_test")
def svm_t():
    train_data_x = [[0, 0], [1, 1]]
    train_data_y = [0, 1]
    test_input = [[2., 2.]]
    result = svm_test(train_data_x, train_data_y, test_input)
    output = "Given train data " + str(train_data_x) + " " + str(train_data_y) + ", input " + str(test_input) + " is classified as " + str(result)
    return output
