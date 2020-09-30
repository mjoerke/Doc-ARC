###
# Some logic from https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO/blob/master/agg.py
##

import os, glob, itertools, sys, re

def generate_collection(ontonotes_dir, output_dir, tag="train"):
	results = itertools.chain.from_iterable(glob.iglob(os.path.join(root,'*.gold_conll'))
											   for root, dirs, files in os.walk('./conll-formatted-ontonotes-5.0/data/'+tag))

	try:
		os.makedirs("%s/%s" % (output_dir, tag))
	except:
		pass

	for result in results:
		
		orig=result
		result=re.sub("./conll-formatted-ontonotes-5.0/data/%s/data/english/annotations/" % tag, "", result)
		result=re.sub("\.gold.conll$", "", result)
		ontonotesFile="%s/data/files/data/english/annotations/%s.name" % (ontonotes_dir, result)

		if os.path.exists(ontonotesFile):

			print(ontonotesFile, os.path.exists(ontonotesFile))
			base=result.split("/")[-1]
			outFile="%s/%s/%s.tsv" % (output_dir, tag, base)

			words=0
			with open(orig) as file:
				for line in file:
					if line.startswith("#"):
						continue
					cols=re.split("\s+", line.rstrip())
					if len(cols) > 10:
						words+=1

			print(tag, words)
			if words < 1000:
				print("skipping")
				continue

			out=open(outFile, "w", encoding="utf-8")

			with open(orig) as file:
				flag=None
				for line in file:
					if line.startswith("#"):
						continue
					cols=re.split("\s+", line.rstrip())

					if len(cols) > 10:
						word=cols[3]
						ori_ner=cols[10]

						if ori_ner == "*":
							if flag==None:
								ner = "O"
							else:
								ner = "I-" + flag
						elif ori_ner == "*)":
							ner = "I-" + flag
							flag = None
						elif ori_ner.startswith("(") and ori_ner.endswith("*") and len(ori_ner)>2:
							flag = ori_ner[1:-1]
							ner = "B-" + flag
						elif ori_ner.startswith("(") and ori_ner.endswith(")") and len(ori_ner)>2 and flag == None:
							ner = "B-" + ori_ner[1:-1]
							flag=None

						out.write("%s\t%s\n" % (word, ner))

					else:
						out.write("\n")

			out.close()

# cd OntoNotes-5.0-NER-BIO
# python3 create_ontonotes_data.py ../ontonotes-release-5.0 outdir

ontonotes_dir=sys.argv[1]
outdir=sys.argv[2]
generate_collection(ontonotes_dir, outdir, tag="train")
generate_collection(ontonotes_dir, outdir, tag="development")
generate_collection(ontonotes_dir, outdir, tag="test")

