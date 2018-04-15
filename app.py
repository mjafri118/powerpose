from flask import Flask, flash, redirect, render_template, request, session


app= Flask(__name__)

# Databse Usage
#db = SQL("sqlite:///database.db")

#Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['GET', 'POST'])
def api():
    if request.method == "GET":
        test = request.args.get('test')
        print(test)
        return '''<h1>The test value is: {}</h1>'''.format(test)

if __name__ == '__main__':
    app.run(debug=True)
