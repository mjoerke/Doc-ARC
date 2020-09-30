
# Filter original GENIA files to extract just those articles we have pdfs for.
# Save each as its own file.

# python filterGENIA.py pdf_list.train.txt original/Genia4ERtask2.iob2 train
# python filterGENIA.py pdf_list.test.txt original/Genia4EReval2.iob2 test

import sys, os, re

def read_ids(filename):
	valid={}
	with open(filename) as file:
		for line in file:
			idd=line.rstrip()
			if len(idd) > 0:
				valid[idd]=1
	return valid


def proc(filename, outputDir, valid):

	try:
		os.makedirs("%s" % outputDir)
	except:
		pass

	docs={}
	with open(filename) as file:
		idd=None
		for line in file:

			if line.startswith("###MEDLINE:"):
				idd=re.sub("###MEDLINE:", "", line.rstrip())
				print(idd)
				docs[idd]=[]

			cols=line.rstrip().split("\t")
			if len(cols) < 2 and len(docs[idd]) == 0:
				continue

			docs[idd].append(line.rstrip())


	for idd in docs:
		if idd in valid:
			with open(os.path.join(outputDir, "%s.tsv" % idd), "w") as out:
				for col in docs[idd]:
					out.write("%s\n" % col)


if __name__ == "__main__":
	idFile=sys.argv[1]
	originalFile=sys.argv[2]
	outputDir=sys.argv[3]
	valid=read_ids(idFile)

	proc(originalFile, outputDir, valid)

