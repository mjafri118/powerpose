from flask import Flask


app= Flask(__name__)

# Databse Usage
#db = SQL("sqlite:///database.db")
# https://progblog.io/How-to-deploy-a-Flask-App-to-Heroku/

#Ensure responses aren't cached
# @app.after_request
# def after_request(response):
#     response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#     response.headers["Expires"] = 0
#     response.headers["Pragma"] = "no-cache"
#     return response

@app.route('/')
def index():
    return "<h1>I'ts working!</h1>"

@app.route('/api', methods=['GET', 'POST'])
def api():
    if request.method == "GET":
        test = request.args.get('test')
        print(test)
        return '''<h1>The test value is: {}</h1>'''.format(test)

# if __name__ == '__main__':
#     #app.run(debug=True)
#     app.run(debug = True, host='127.0.0.1', port=1424)
