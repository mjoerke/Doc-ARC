
# Fix line-break hyphenization for OCR and tokenize text (one sentence per line)
# python OCR_postprocess.py 20563930.txt

import sys, nltk, re

vocab={}

def read_text(filename):
	lines=[]
	with open(filename) as file:
		for line in file:
			tokens=line.rstrip().split(" ")
			lines.append(tokens)
	return lines

def read_vocab(filename, lines):

	with open(filename) as file:
		for line in file:
			vocab[line.rstrip().lower()]=1
	
	for line in lines:
		for word in line:
			if not word.endswith("-"):
				vocab[word.lower()]=1
	

def proc(filename):

	tokenized_lines=read_text(filename)
	read_vocab("/usr/share/dict/words", tokenized_lines)
	
	tokens=[]
	previousLineHyphenMatch=False

	for idx,words in enumerate(tokenized_lines):
		flag=False
		
		# if line ends in hyphen
		if len(words) > 0 and words[-1].endswith("-") and idx < len(tokenized_lines)-1:
			nextwords=tokenized_lines[idx+1]
			if len(nextwords) > 0:
				first=nextwords[0]
				candidate="%s%s" % (re.sub("-$", "", words[-1]), first)
				candidate2="%s-%s" % (re.sub("-$", "", words[-1]), first)
				
				# check if candidate word exists in dictionary
				if candidate.lower() in vocab:
					# if so, replace the fragment with the full word
					words[-1]=candidate
				else:
					words[-1]=candidate2

					# and keep a flag to we know to drop the first word of the next line
				flag=True

			   
		if previousLineHyphenMatch:
			tokens.append(words[1:])
		else:
			tokens.append(words)
		
		previousLineHyphenMatch = True if flag else False

	all_tokens=[]
	for line in tokens:
		all_tokens.extend(line)
	doc=' '.join(all_tokens)

	for sent in nltk.sent_tokenize(doc):
		print (' '.join(nltk.word_tokenize(sent)))

proc(sys.argv[1])
