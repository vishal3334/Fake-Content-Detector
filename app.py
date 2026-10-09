import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from text_analyzer import predict_text
from image_analyzer import analyze_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
UPLOAD_DIR = "uploads"
ALLOWED = {"png", "jpg", "jpeg", "webp"}
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, kind = None, None, None

    if request.method == "POST":
        kind = request.form.get("kind")

        if kind == "text":
            text = request.form.get("text", "").strip()
            if len(text.split()) < 5:
                error = "Please enter at least 5 words of text."
            else:
                result = predict_text(text)

        elif kind == "image":
            file = request.files.get("image")
            if not file or file.filename == "":
                error = "Please choose an image file."
            else:
                ext = file.filename.rsplit(".", 1)[-1].lower()
                if ext not in ALLOWED:
                    error = "Only PNG, JPG, JPEG or WEBP files are allowed."
                else:
                    path = os.path.join(UPLOAD_DIR, secure_filename(file.filename))
                    file.save(path)
                    result = analyze_image(path, ela_output_path=os.path.join("static", "ela_result.png"))

    return render_template("index.html", result=result, error=error, kind=kind)


if __name__ == "__main__":
    app.run(debug=True)
