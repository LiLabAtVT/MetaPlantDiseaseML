#!/usr/bin/env python3.7

####################################################################################
##this script is to generate seqs and labels that will be ready for the CNN training
def load_data(fname):
    print("==========Splitting sequences and labels==========")
    seqs = []
    labels = []
    f = open(fname)
    for line in f:
      if(line.startswith('>')):
         line_no_comma = line.replace(",","")
         line_no_start = line_no_comma.replace(">","")
         line_no_nwline = line_no_start.replace("\n","")
         labels.append(line_no_nwline)
      else:
         seqnc = line
         seq_no_nwline = seqnc.replace("\n","")
         seqs.append(seq_no_nwline)
    f.close()
    return seqs, labels
