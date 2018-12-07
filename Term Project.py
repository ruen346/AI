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
print("x6과 x8을 기준으로 학습을 하여 분석한 결과를 보여드리겠습니다. ")
csv = []
with open('train.csv', 'r', encoding='utf-8') as fp:
	# 한 줄씩 읽어 들이기
	for line in fp:
		line = line.strip()		# 줄바꿈 제거
		cols = line.split(',') 	# 쉼표로 자르기
		# 문자열 데이터를 숫자로 변환하기
		fn = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
		cols = list(map(fn, cols))
		csv.append(cols)


# 가장 앞 줄의 헤더 제거
del csv[0]

# 데이터 셔플하기(섞기) --- (※2)
random.shuffle(csv)

# 학습 전용 데이터와 테스트 전용 데이터 분할하기(3:1 비율) --- (※3)
total_len = len(csv)
train_len = int(total_len * 9 / 10)
train_data = []
train_label = []
test_data = []
test_label = []

for i in range(total_len):
	data = csv[i][0:9]
	label = csv[i][5]
	if i < train_len:
		train_data.append(data)
		train_label.append(label)
	else:
		test_data.append(data)
		test_label.append(label)

# 데이터를 학습시키고 예측하기 --- (※4)
clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

# 정답률 구하기 --- (※5)
ac_score = metrics.accuracy_score(test_label, pre)
print("X6, 9:1로 분석한 결과 =", ac_score)
"""