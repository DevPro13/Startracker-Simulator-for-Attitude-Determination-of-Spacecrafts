from flask import Flask,render_template,request,send_file,request,abort
import sys
import os
sys.path.append('../ImageGenerator')
from imageGen import GenerateStarImage
StarImagesParameters=dict()
app=Flask(__name__)
@app.route('/',methods=['GET', 'POST']) 
def home(): 
    if request.method=='POST':
        StarImagesParameters["RA"]=str(request.form.get('RA'))
        StarImagesParameters["DEC"]=str(request.form.get('DEC'))
        StarImagesParameters["ROLL"]=str(request.form.get('ROLL'))
        StarImagesParameters["FOVx"]=str(request.form.get('FOVx'))
        StarImagesParameters["FOVy"]=str(request.form.get('FOVy'))
        StarImagesParameters["RES_X"]=str(request.form.get('RES_X'))
        StarImagesParameters["RES_Y"]=str(request.form.get('RES_Y'))
        StarImagesParameters["Mv"]=str(request.form.get('mag'))
        StarImagesParameters["sigma"]=str(request.form.get('sigma'))
        StarImagesParameters["noise"]=str(request.form.get('noise'))
        print(StarImagesParameters)
        GenerateImage=GenerateStarImage(StarImagesParameters)
        catalog=GenerateImage.generateCatalog()
        filename=GenerateImage.generateImage(catalog)
        description=GenerateImage.description()
        return render_template("index.html",description=description,path=filename) 
    #path='../Media/ra12.11316809533822_de-3.5483542856278105_roll43.909061625097_FOV16.4_GEMINI_FOV16.4.png'
    return render_template("index.html")
@app.route('/get_image')
def get_image():
    # Get the path parameter from the query string
    image_path = request.args.get('path')
    # Check if the file exists
    if not os.path.exists(image_path):
        abort(404, "Image file not found")
    return send_file(image_path, mimetype='image/png')
if __name__=='__main__':
    app.run(debug=True)