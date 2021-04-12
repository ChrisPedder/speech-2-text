from flask import Flask, render_template, request
from STT_model import STT_model

app = Flask(__name__)
model = STT_model()

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def convert_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(f.filename)
      transcription = model.convert(f.filename)
      return {"transcription": transcription}

if __name__ == '__main__':
   app.run(debug = True)
