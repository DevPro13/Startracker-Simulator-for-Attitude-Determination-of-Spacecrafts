from flask import Flask,render_template,request
StarImagesParameters=dict()
app=Flask(__name__)
@app.route('/',methods=['GET', 'POST']) 
def home(): 
    if request.method=='POST':
        StarImagesParameters["RA"]=request.form.get('RA')
        StarImagesParameters["DEC"]=request.form.get('DEC')
        StarImagesParameters["ROLL"]=request.form.get('ROLL')
        StarImagesParameters["FOVx"]=request.form.get('FOVx')
        StarImagesParameters["FOVy"]=request.form.get('FOVy')
        StarImagesParameters["RES_X"]=request.form.get('RES_X')
        StarImagesParameters["RES_Y"]=request.form.get('RES_Y')
        StarImagesParameters["Mv"]=request.form.get('mag')
        StarImagesParameters["sigma"]=request.form.get('sigma')
        StarImagesParameters["noise"]=request.form.get('noise')
        print(StarImagesParameters)
        return render_template("index.html") 
    return render_template("index.html") 
if __name__=='__main__':
    app.run(debug=True)