from flask import Flask, render_template, jsonify
from utils import api_keys, uploaded_files
from llamaindex import main
from langchain_folder import agent
from crewai_local import trip_crew

app = Flask(__name__)


@app.route("/")
def hello_page():
    return render_template('home.html')



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)