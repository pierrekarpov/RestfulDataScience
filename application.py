from flask import Flask, request, render_template
from computation import svm_test, make_prediction
from renderers import render_predict_form, render_prediction

app = Flask(__name__)
# $ source restful_data_science/bin/activate
# $ FLASK_APP=application.py flask run

@app.route("/")
@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)


@app.route("/svm_test")
def svm_t():
    train_data_x = [[0, 0], [1, 1]]
    train_data_y = [0, 1]
    test_input = [[2., 2.]]
    result = svm_test(train_data_x, train_data_y, test_input)
    output = "Given train data " + str(train_data_x) + " " + str(train_data_y) + ", input " + str(test_input) + " is classified as " + str(result)
    return output
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
