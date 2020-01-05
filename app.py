import flask
from flask import render_template
import pickle
import sklearn
import random
import pandas

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        temp = 1
        with open('model.pickle', 'rb') as fh:
            loaded_model = pickle.load(fh)

        with open('scaler.pickle', 'rb') as fh:
            scaler = pickle.load(fh)

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

        def generate_random_values(carat_unique, cut_unique, color_unique, clarity_unique, depth_unique, table_unique,
                                   x_unique, y_unique, z_unique):

            df = pd.DataFrame({
                'carat': random.choice(carat_unique),
                'cut': random.choice(cut_unique),
                'color': random.choice(color_unique),
                'clarity': random.choice(clarity_unique),
                'depth': random.choice(depth_unique),
                'table': random.choice(table_unique),
                'x': random.choice(x_unique),
                'y': random.choice(y_unique),
                'z': [random.choice(z_unique)]})

            return df

        if 'Send' in flask.request.form:
            dic = flask.request.form.to_dict(flat=False)
            del dic['Send']
            df = pd.DataFrame(dic)
            temp = loaded_model.predict(transform_data(df, scaler, unique))

        elif 'Generate' in flask.request.form:
            df = generate_random_values(*unique.values())
            dic = df.to_dict('list')
            temp = loaded_model.predict(transform_data(df, scaler, unique))

        return render_template('main_print.html', data=dic, result=temp)


if __name__ == '__main__':
    app.run()