from pyvi import ViTokenizer
import re
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from flask import Flask, render_template, request
import os

base_path = os.path.dirname(os.path.abspath(__file__))

# giai nen cac doi tuong
clf = pickle.load(open(os.path.join(base_path, 'Version3/Colab/NB_ChatBot_model.pkl'), 'rb'))
# clf = pickle.load(open('Colab/NB_ChatBot_model.pkl', 'rb'))
vocabulary_to_load = pickle.load(open(os.path.join(base_path, 'Version3/Colab/vocab.pkl'), 'rb'))
le = pickle.load(open(os.path.join(base_path, 'Version3/Colab/decode_label.pkl'), 'rb'))

# app = Flask(__name__)  # khoi tao flask
app = Flask(__name__, template_folder="Version3/templates", static_folder="Version3/static")



@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def chatbot_response():
    if request.method == "POST":
        message = request.form.get("msg")
        ok = prediction(message)
    return ok


def tienxuly(document):
    document = ViTokenizer.tokenize(document)
    # đưa về lower
    document = document.lower()
    # xóa các ký tự không cần thiết
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', document)
    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()
    return document


stopword = ["nghành", "gì", "là", "trường", "năm"]


def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)


def prediction(input):
    ngram_size = 1
    # kích thước của các 'ngram' sẽ được sử dụng khi tạo vector từ văn bản. Trong trường hợp này, chỉ sử dụng unigrams (từ đơn)
    loaded_vectorizer = CountVectorizer(ngram_range=(ngram_size, ngram_size), min_df=1,
                                        vocabulary=vocabulary_to_load)
    loaded_vectorizer._validate_vocabulary()
    a = tienxuly(input)

    input1 = remove_stopwords(a)
    vect = loaded_vectorizer.transform([input1]).toarray()
    predict = clf.predict(vect)
    predict = le.inverse_transform(predict)[0]

    if predict == "noanswer":
        predict = "xin lỗi bạn, câu này tôi không biết trả lời như thế nào. Bạn vui lòng liện hệ theo số điện thoại 123456 để được tư vấn trực tiếp"

    return predict


if __name__ == "__main__":
    app.run(debug=True)

# stopword không mang ý nghĩa đặc biệt trong việc hiểu nghĩa của văn bản
# việc loại bỏ stopwords có thể giúp giảm kích thước của dữ liệu và tăng tốc quá trình xử lý