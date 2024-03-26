from flask import Flask, render_template, jsonify, request
from utils import api_keys, uploaded_files
from llamaindex import main
from langchain_folder import agent
from crewai_local import trip_crew

app = Flask(__name__)

path_to_directory=[]
api_key_dict = {}

@app.route("/")
def welcome_page():
    return render_template('home.html')

@app.route('/agents/file_api_key')
def api():
    return render_template('file_apikey.html')

@app.route('/agents/langchain')
def langchain_main():
    return render_template('langchain.html')

@app.route('/agents/llamaindex')
def llamaindex_main():
    return render_template('llamaindex.html')

@app.route('/agents/crewai')
def crewai_main():
    return render_template('crewai.html')


@app.route("/agents/file_api_key", methods = ['post'])
def upload_api_page():
    openai_api_key = request.form.get('openai_api_key')
    serper_api_key = request.form.get('serper_api_key')
    browserless_api_key = request.form.get('browserless_api_key')

    file_one = request.files.get('file_one')
    file_two = request.files.get('file_two')

    files = uploaded_files(file_one, file_two)
    path_to_directory.extend(files)

    keys = api_keys(openai_api_key, serper_api_key, browserless_api_key)

    for k, v in keys.items():
        api_key_dict[k]=v

    return render_template('api_thankyou.html')


@app.route("/agents/langchain", methods = ['post'])
def langchain_page():
    data = request.form
    model = agent.langchain_agent(api_key_dict['OPENAI_API_KEY'],
                                  api_key_dict['SERPER_API_KEY'],
                                  path_to_directory[1],
                                  path_to_directory[2])
    response = model.generate_langchain(data['text_input'])

    return render_template('langchain_result.html',
                           generated = response)


@app.route("/agents/llamaindex", methods=['post'])
def llama_index_page():
    data = request.form
    model = main.generate_llamaindex(data['text_input'],
                                      path_to_directory[0],
                                      api_key_dict["OPENAI_API_KEY"])
    
    return render_template('llamaindex_result.html',
                           generated = model)


@app.route("/agents/crewai", methods=['post'])
def crewai_page():
    data = request.form
    model = trip_crew.TripCrew(data['location'],
                               data['cities'],
                               data['date_range'],
                               data['interests'],
                               api_key_dict['OPENAI_API_KEY'])

    response = model.run()

    return render_template('crewai_result.html',
                           generated = response)











if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)