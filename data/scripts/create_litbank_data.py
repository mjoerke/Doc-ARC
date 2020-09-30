import sys, glob, re, os

def process(data, out):
	sent=[]
	for cols in data:
		sent.append(cols[0])

	labels=["O"]*len(sent)
	for i in range(len(data[0])-1, 0, -1):
		
		for j in range(len(data)):
			if labels[j] == "O":
				labels[j]=data[j][i]

	for i in range(len(sent)):
		out.write("%s\t%s\n" % (sent[i], labels[i]))
	out.write("\n")

def flatten(filename, outputFile):

	with open(filename) as file:
		data=[]
		for line in file:

			cols=line.rstrip().split("\t")
			if len(cols) < 2:
				process(data, outputFile)
				data=[]
				continue

			data.append(cols)


def proc(folder, outputFolder):

	try:
		os.makedirs(outputFolder)
	except:
		pass

	files = glob.iglob(os.path.join(folder,'*.tsv'))

	for filename in files:
		idd=filename.split("/")[-1]
		idd=re.sub("\.tsv$", "", idd)
		print(idd)
		outputFile="%s/%s.tsv" % (outputFolder, idd)
		out=open(outputFile, "w", encoding="utf-8")

		flatten(filename, out)		
		out.close()	

proc(sys.argv[1], sys.argv[2])


