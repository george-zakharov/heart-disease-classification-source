import flask
from flask import render_template
import pickle
import random
import pandas as pd

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        return render_template('main.html')
        temp = 1
        with open('model.pickle', 'rb') as fh:
            loaded_model = pickle.load(fh)

        with open('unique.pickle', 'rb') as fh:
            unique = pickle.load(fh)

        def transform_data(data, scaler, unique):
            dummies = pd.get_dummies(unique['cut_unique'])
            CCut = dummies.loc[dummies[data['cut'].values[0]] == 1]
            dummies = pd.get_dummies(unique['color_unique'])
            CColor = dummies.loc[dummies[data['color'].values[0]] == 1]
            dummies = pd.get_dummies(unique['clarity_unique'])
            CClarity = dummies.loc[dummies[data['clarity'].values[0]] == 1]

            df = pd.concat(
                [data, CCut.reset_index(drop=True), CColor.reset_index(drop=True), CClarity.reset_index(drop=True)],
                axis=1)
            df = df.drop(['cut', 'color', 'clarity'], axis=1)
            df = scaler.transform(df.values.reshape(df.shape[0], -1))

            return df

        def generate_random_values(oldpeak, thalach, age, trestbps, chol, thal, cp, ca, exang, slope):
            df = pd.DataFrame({
                'oldpeak': random.choice(oldpeak),
                'thalach': random.choice(thalach),
                'age': random.choice(age),
                'trestbps': random.choice(trestbps),
                'chol': random.choice(chol),
                'thal': random.choice(thal),
                'cp': random.choice(cp),
                'ca': random.choice(ca),
                'exang': random.choice(exang),
                'slope': random.choice(slope)
            }, index=[0])

            return df

        if 'Send' in flask.request.form:
            dic = flask.request.form.to_dict(flat=False)
            del dic['Send']
            df = pd.DataFrame(dic)
            temp = loaded_model.predict(df)

        elif 'Generate' in flask.request.form:
            df = generate_random_values(*unique.values())
            dic = df.to_dict('list')
            temp = loaded_model.predict(df)

        return render_template('main_result.html', data=dic, result=temp)


if __name__ == '__main__':
    app.run()
