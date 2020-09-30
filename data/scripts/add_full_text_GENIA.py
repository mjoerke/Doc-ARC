import sys, re
import glob, itertools, os

def read_book(filename):
	lastSid=None
	sents=[]
	with open(filename) as file:

		for idx, line in enumerate(file):

			sent=line.rstrip().split(" ")
			sents.append(sent)
	
	return sents


def proc_gold(filename, sents, outFile):
	out=open(outFile, "w", encoding="utf-8")

	with open(filename) as file:
		for line in file:
			out.write("%s\n" % line.rstrip())

	for sent in sents:
		for word in sent:
			out.write("%s\t%s\n" % (word, "NULL"))
		out.write("\n")

	out.close()

def proc(orig_folder, out_folder):
	
	try:
		os.makedirs("%s" % out_folder)
	except:
		pass


	filenames = glob.iglob(os.path.join(orig_folder,'*.tsv'))
	for goldFile in filenames:
		idd=goldFile.split("/")[-1]
		idd=re.sub("\.tsv$", "", idd)

		bookFile="articles/processed_texts/%s.txt" % idd
		outFile="%s/%s.tsv" % (out_folder, idd)
		sents=read_book(bookFile)
		proc_gold(goldFile, sents, outFile)


orig_folder=sys.argv[1]
out_folder=sys.argv[2]
proc(orig_folder, out_folder)