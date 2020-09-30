import sys, glob, itertools, os

def proc(folder):
	tags={}
	filenames = itertools.chain.from_iterable(glob.iglob(os.path.join(root,'*.tsv')) for root, dirs, files in os.walk(folder))

	for filename in filenames:
		with open(filename) as file:
			for line in file:
				cols=line.rstrip().split("\t")

				if len(cols) > 1:
					tag=cols[1]
					if tag not in tags:
						tags[tag]=len(tags)

	for tag in tags:
		print("%s\t%s" % (tag, tags[tag]))


proc(sys.argv[1])