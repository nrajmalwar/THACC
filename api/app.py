#pipenv run python api/app.py

from flask import Flask
import timeline_prediction

app = Flask(__name__)

@app.route('/')
def timeline():
    timeline_prediction.generate_timeline()
    return "<h1>Find the complete financial history in data folder</h1>"

if __name__ == '__main__':
    app.run(debug=True)
