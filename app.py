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
            response = flask.request.form.to_dict(flat=False)
            if 'thal' not in response:
                response['thal'] = '0'

            temp = loaded_model.predict([[
                float(response['thalach'][0]), float(response['oldpeak'][0]), float(response['age'][0]),
                float(response['chol'][0]), float(response['trestbps'][0]), int(response['thal'][0]),
                int(response['cp'][0]), int(response['ca'][0]), int(response['exang'][0]), int(response['slope'][0])
            ]])
            if temp is not None:
                if temp[0] == 0:
                    temp = 'No disease, predicted class - 0'
                else:
                    temp = 'There is a disease, predicted class - 1'

        return render_template('main.html', result=temp)


if __name__ == '__main__':
    app.run()
