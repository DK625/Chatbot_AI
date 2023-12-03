from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pyvi import ViTokenizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import nltk
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

data = pd.read_excel('Data2.xlsx')
X = data["question"]
Y = data["answer"]
print("thuoc tinh dieu kien")
print(X)
print("thuoc tinh can du doan")
print(Y)

# Tiep theo chung ta se ma hoa Y sao cho no hop ly
le = preprocessing.LabelEncoder()
le.fit(Y)

list_label = list(le.classes_)

print(list_label)
print(len(list_label))

label = le.transform(Y)
print(label)


def tienxuly(document):
    document = ViTokenizer.tokenize(document)
    # đưa về lower
    document = document.lower()
    # xóa các ký tự không cần thiết
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', document)
    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()
    return document


for i in range(0, X.count()):
    X[i] = tienxuly(X[i])

# chung ta se loai bo stop word trong van ban


# bay h dau tien minh muon biet cac stop word o dau va cho nao

# tokens = [t for t in text.split()]
tokens = []

for i in range(0, X.count()):
    for j in X[i].split():
        tokens.append(j)

freq = nltk.FreqDist(tokens)
freq.plot(20, cumulative=False)

# nhu vay chung ta biet mot so tu xuat hien xuat hien thuong xuyen va no se anh huong toi mo hinh can du doan
# chung ta se loai bo chung de cho model co do chinh xac cao hon

stopword = ["nghành", "gì", "là", "trường", "năm"]


def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)


print(X[4])
demo = remove_stopwords(X[4])

for i in range(0, X.count()):
    X[i] = remove_stopwords(X[i])

# buoc tiep theo chung ta se xay dung bo tu dien cho may hoc
vectorizer = CountVectorizer()


def transform(data):
    data = list(data)
    return vectorizer.fit_transform(data).todense()


# data1 = transform(X)
data1 = np.asarray(transform(X))

print(data1)

# chia du lieu ra lam 2 phan
X_train, X_test, Y_train, Y_test = train_test_split(data1, label, test_size=0.1, random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# khoi tao mo hinh
