import flask
from flask import render_template
import pickle
import pandas as pd

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        temp = 0
        with open('model.pkl', 'rb') as fh:
            loaded_model = pickle.load(fh)

        if 'Send' in flask.request.form:
            dic = flask.request.form.to_dict(flat=False)
            del dic['Send']
            df = pd.DataFrame(dic)
            temp = loaded_model.predict(df)

        return render_template('main.html', result=dic)


if __name__ == '__main__':
    app.run()
