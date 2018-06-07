import sqlite3 as sql

db_name = "restful_data_science.db"

# TODO: need to check if no record with that filename exist
def insertModel(classifier_type, feature_count, file_name):
    con = sql.connect(db_name)
    cur = con.cursor()
    cur.execute("INSERT INTO models (classifier_type, feature_count, file_name) VALUES (?, ?, ?)", (classifier_type, feature_count, file_name))
    id = cur.lastrowid
    con.commit()
    con.close()

    return id

def retrieveModels():
    con = sql.connect(db_name)
    cur = con.cursor()
    cur.execute("SELECT id, classifier_type, feature_count, file_name FROM models")
    models = cur.fetchall()
    con.close()
    return models

def retrieveModel(id):
    con = sql.connect(db_name)
    cur = con.cursor()
    cur.execute("SELECT id, classifier_type, feature_count, file_name FROM models WHERE id = " + str(id))
    model = cur.fetchall()
    con.close()
    return model
