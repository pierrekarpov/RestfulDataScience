# RestfulDataScience

A Flask project to do data science through REST API

## Getting Started

Clone this project on your machine to have the code necessary to run this project locally.

### Prerequisites

You will need Python, pip, and virtualenv to get this project started.
Install Python and pip here:
[Python](https://www.python.org/downloads/)
[pip](https://pip.pypa.io/en/stable/installing/)

Then use pip to install virtualenv
```
$ pip install virtualenv
```

### Installing

To install the necessary packages, you can simply run

```
$ pip install -r requirements.txt
```

However, it is recommended to created an environment so that those imports are specific to this project
Create your environment
```
$ virtualenv <YOUR_ENV_NAME>
```

Activate your environment
```
$ source <YOUR_ENV_NAME>/bin/activate
```

Install packages
```
$ (YOUR_ENV_NAME) pip install -r requirements.txt
```

To deactivate your environment
```
$ (YOUR_ENV_NAME) deactivate
```

To lauch the Flask app, run
```
FLASK_APP=application.py flask run
```

Then head over to (http://localhost:5000/predict) to test different classifiers over the [wine](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) data provided by [scikit-learn](http://scikit-learn.org/stable/)

## Running the tests

[TODO]

### Break down into end to end tests

[TODO]

### And coding style tests

[TODO]

## Deployment

[TODO]

## Built With

[TODO]

## Contributing

[TODO]

## Versioning

[TODO]

## Authors

[TODO]

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

[TODO]
