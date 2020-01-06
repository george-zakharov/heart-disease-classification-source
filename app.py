import flask
from flask import render_template
import pickle

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    temp = None
    if flask.request.method == 'GET':
        return render_template('main.html', result=temp)

    if flask.request.method == 'POST':
        with open('model.pkl', 'rb') as fh:
            loaded_model = pickle.load(fh)

        if 'Send' in flask.request.form:
            dic = flask.request.form.to_dict(flat=False)
            if 'thal' not in dic:
                # And yes, we need this line to fix some input issues, cuz this key can randomly be empty
                dic['thal'] = '0'

            temp = loaded_model.predict([[
                float(dic['thalach'][0]), float(dic['oldpeak'][0]), float(dic['age'][0]), float(dic['chol'][0]),
                float(dic['trestbps'][0]), int(dic['thal'][0]), int(dic['cp'][0]), int(dic['ca'][0]),
                int(dic['exang'][0]), int(dic['slope'][0])
            ]])
            if temp is not None:
                temp = str(temp[0])

        return render_template('main.html', result=temp)


if __name__ == '__main__':
    app.run()
