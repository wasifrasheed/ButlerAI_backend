from blue import app

if __name__ == "__main__":
    #For local host
    #app.run(debug = True)

    #For assigning port no and  host
    app.run(host="0.0.0.0",port=8008,debug=True)

    #For stopping defualt reloading of server
    #app.run(host="0.0.0.0",port=5000,use_reloader=False)

    #To handle with multiple request
    #app.run(host="0.0.0.0", port=8000,threaded=True)

    #Best approach 
   # app.run(host="0.0.0.0", port=5000, use_reloader=False,threaded=True)



