import urllib.request
import urllib.request as req
from sklearn import svm, metrics
import random, re
import xlrd
import unicodecsv
import gzip, os, os.path
import struct
from sklearn import model_selection, svm, metrics




savepath = "./mnist"
baseurl = "http://yann.lecun.com/exdb/mnist"
files = [
"train-images-idx3-ubyte.gz",
"train-labels-idx1-ubyte.gz",
"t10k-images-idx3-ubyte.gz",
"t10k-labels-idx1-ubyte.gz"]

# 다운로드
if not os.path.exists(savepath): os.mkdir(savepath)
for f in files:
	url = baseurl + "/" + f
	loc = savepath + "/" + f
	print("download:", url)
	if not os.path.exists(loc):
		req.urlretrieve(url, loc)

# GZip 압축 해제
for f in files:
	gz_file = savepath + "/" + f
	raw_file = savepath + "/" + f.replace(".gz", "")
	print("gzip:", f)
	with gzip.open(gz_file, "rb") as fp:
		body = fp.read()
		with open(raw_file, "wb") as w:
			w.write(body)
print("ok")

def to_csv(name, maxdata):
	# 레이블 파일과 이미지 파일 열기
	lbl_f = open("./mnist/"+name+"-labels-idx1-ubyte", "rb")
	img_f = open("./mnist/"+name+"-images-idx3-ubyte", "rb")
	csv_f = open("./mnist/"+name+".csv", "w", encoding="utf-8")
	# 헤더 정보 읽기 --- (※1)
	mag, lbl_count = struct.unpack(">II", lbl_f.read(8))
	mag, img_count = struct.unpack(">II", img_f.read(8))
	rows, cols = struct.unpack(">II", img_f.read(8))
	pixels = rows * cols
	# 이미지 데이터를 읽고 CSV로 저장하기 --- (※2)
	res = []
	for idx in range(lbl_count):
		if idx > maxdata: break
		label = struct.unpack("B", lbl_f.read(1))[0]
		bdata = img_f.read(pixels)
		sdata = list(map(lambda n: str(n), bdata))
		csv_f.write(str(label)+",")
		csv_f.write(",".join(sdata)+"\r\n")
		# 잘 저장됐는지 이미지 파일로 저장해서 테스트하기 -- (※3)
		if idx < 10:
			s = "P2 28 28 255\n"
			s += " ".join(sdata)
			iname = "./mnist/{0}-{1}-{2}.pgm".format(name, idx, label)
			with open(iname, "w", encoding="utf-8") as f:
				f.write(s)
	csv_f.close()
	lbl_f.close()
	img_f.close()


# 결과를 파일로 출력하기 --- (※4)
to_csv("train", 1000)
to_csv("t10k", 500)


"""
def load_csv(fname):
	labels = []
	images = []
	with open(fname, "r") as f:
		for line in f:
			cols = line.split(",")
			if len(cols) < 2: continue
			labels.append(int(cols.pop(0)))
			vals = list(map(lambda n: int(n) / 256, cols))
			images.append(vals)
	return {"labels":labels, "images":images}

data = load_csv("./mnist/train.csv")
test = load_csv("./mnist/t10k.csv")

# 학습하기 --- (※2)
clf = svm.SVC()
clf.fit(data["images"], data["labels"])

# 예측하기 --- (※3)
predict = clf.predict(test["images"])

# 결과 확인하기 --- (※4)
ac_score = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"], predict)
print("정답률 =", ac_score)
print("리포트 =")
print(cl_report)
"""
def load_csv(fname):
	labels = []
	images = []
	with open(fname, "r") as f:
		for line in f:
			cols = line.split(",")
			if len(cols) < 2: continue
			labels.append(int(cols.pop(0)))
			vals = list(map(lambda n: int(n) / 256, cols))
			images.append(vals)
	return {"labels":labels, "images":images}

total = load_csv("./mnist/train.csv")

#9 : 1 비율로 나누기
print("9 : 1 비율로 나누는 중입니다.")
total_len = len(total)
train_len = int(total_len * 9 / 10)
print(type(total))
test = total
data = total

# 학습하기 --- (※2)
clf = svm.SVC()
print(type(clf))
clf.fit(data["images"], data["labels"])

# 예측하기 --- (※3)
predict = clf.predict(test["images"])

# 결과 확인하기 --- (※4)
ac_score = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"], predict)
print("정답률 =", ac_score)
print("리포트 =")
print(cl_report)