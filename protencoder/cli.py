"""Console script for protencoder."""
import argparse
import sys
import glob
from Bio import SeqIO
from collections import Counter
from protencoder.onehot import ONEencoder
from protencoder.GOencoder import GOencoder
from protencoder.kmerHz import protKmers
from protencoder.coMatrix import AAcomptability
from protencoder.protVec import protvec


def devide(seqPath, chopSize, outPrefix):
    fasta_sequences = SeqIO.parse(seqPath, 'fasta')
    x = 0
    newFasta = list()
    part = 1
    for seq in fasta_sequences:
        newFasta.append(seq)
        x += 1
        if x == chopSize:
            SeqIO.write(newFasta, outPrefix+"_part"+str(part)+".fasta",
                        'fasta')
            newFasta = list()
            x = 0
            part += 1
    if x != 0:
        SeqIO.write(newFasta, outPrefix+"_part"+str(part)+".fasta",
                    'fasta')


def create_filter(collection, numFreqGO, outPrefix):
    seqDict = {'F': list(), 'P': list(), 'C': list()}
    GOAf = open(collection)
    for line in GOAf.readlines():
        line = line.rstrip("\n").split("\t")
        seqDict[line[2]].append(line[1])
    GOAf.close()
    ofile = outPrefix+"_filter_"+str(numFreqGO)
    filterF = open(ofile, 'w')
    counter = Counter(seqDict['F'])
    most_occur = [x[0] for x in counter.most_common(numFreqGO)]
    for GOA in most_occur:
        filterF.write(GOA+"\t"+'F'+"\n")

    counter = Counter(seqDict['P'])
    most_occur = [x[0] for x in counter.most_common(numFreqGO)]
    for GOA in most_occur:
        filterF.write(GOA+"\t"+'P'+"\n")

    counter = Counter(seqDict['C'])
    most_occur = [x[0] for x in counter.most_common(numFreqGO)]
    for GOA in most_occur:
        filterF.write(GOA+"\t"+'C'+"\n")
    filterF.close()
    return ofile


def main():
    """Console script for protencoder."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    filePathParser = parser.add_mutually_exclusive_group(required=True)
    filePathParser.add_argument("-d", "--seqPath",
                                help="file path of the protein sequences")
    filePathParser.add_argument("-g", "--GOfile",
                                help="file path of the GOA of proteins")
    filePathParser.add_argument("-P", "--predictionFile",
                                help="file path pattern of the predictions to\
                                      be decoded")
    parser.add_argument("-M", "--method", default="o",
                        help="protein encoding method; o: (defult)onehot,\
                        k: kmers frequency, c: compatibility matrices")
    parser.add_argument("-e", "--enlargenMode", default='pad',
                        help="mode to enlarge a small protein in method\
                        compatibility matrices; pad(default), resize, tile,\
                        repeat")
    parser.add_argument("-k", "--kmerLength", default="3",
                        help="kmer length in frequency encoder")
    parser.add_argument("-f", "--Protfilter", default="",
                        help="file path of protein keys to be used for\
                        encoding filteration. in case of multiple filters\
                        enter the consensus path followed by *")
    parser.add_argument("-V", "--flatten", default="0",
                        help="whether to flatten protvec or not;\
                        0(default): false, 1: true")
    parser.add_argument("-v", "--PVmodelPath",
                        default="protencoder/data/swissprot-reviewed-protvec.model",
                        help="file of path of a pre-trained protvec model")
    parser.add_argument("-m", "--maxLen", default="2000",
                        help="maximum length of the protein to be encoded;\
                        if -1 m = max protein length, default = 2000")
    parser.add_argument("-p", "--GOpartioned", default="0",
                        help="if annotated keys are divided;\
                        0(default) if False, 1 if True")
    parser.add_argument("-o", "--outPrefix", default="",
                        help="output files prefix")
    parser.add_argument("-s", "--chopSize", default="-1",
                        help="number of sequences to be loaded\
                        into memory per time")
    GOfilterParser = parser.add_mutually_exclusive_group()
    GOfilterParser.add_argument("-F", "--GOfilter", default="",
                                help="file path of GO keys to be used for\
                                encoding filteration")
    GOfilterParser.add_argument("-c", "--collection", default="",
                                help="file path of collected\
                                protein annotations")
    parser.add_argument("-n", "--numFreqGO", default="500", help="number of\
                        the most frequent GO annotions to be considered")
    parser.add_argument("-x", "--dsize", default="500", help="width of\
                        the squared image of protein after compression")
    args = parser.parse_args()

    seqPath = args.seqPath
    GOfile = args.GOfile
    Protfilter = args.Protfilter
    GOfilter = args.GOfilter
    maxLen = int(args.maxLen)
    chopSize = int(args.chopSize)
    GOpartioned = bool(int(args.GOpartioned))
    collection = args.collection
    numFreqGO = int(args.numFreqGO)
    outPrefix = args.outPrefix
    method = args.method
    k = int(args.kmerLength)
    dsize = int(args.dsize)
    flatten = bool(int(args.flatten))
    PVmodel = args.PVmodelPath
    action = args.enlargenMode
    predictionFile = args.predictionFile

    if not (GOfile is None):
        outPrefix = args.outPrefix if args.outPrefix != "" else GOfile
    elif not (seqPath is None):
        outPrefix = args.outPrefix if args.outPrefix != "" else seqPath
    elif not (predictionFile is None):
        outPrefix = args.outPrefix if args.outPrefix != "" else predictionFile

    if collection != "":
        GOfilter = create_filter(collection, numFreqGO, outPrefix)

    if method in "okcp":
        if not (seqPath is None):
            if outPrefix == seqPath:
                outPrefix = seqPath[:-6]
            if method == 'o':
                encoder = ONEencoder(maxLen)
            elif method == 'k':
                encoder = protKmers(k)
            elif method == 'c':
                encoder = AAcomptability(dsize, action)
            elif method == 'p':
                encoder = protvec(PVmodel, flatten)
            if Protfilter != "":
                encoder.load_filter(Protfilter)
            if chopSize == -1:
                if '*' in seqPath:
                    for name in glob.glob(seqPath):
                        encoder.read(name)
                        encoder.encode()
                        encoder.dump(name.rstrip(".fasta"))
                else:
                    encoder.read(seqPath)
                    encoder.encode()
                    encoder.dump(outPrefix)
            else:
                if '*' in seqPath:
                    for name1 in glob.glob(seqPath):
                        devide(name1, chopSize, name1.rstrip(".fasta"))
                        name1 = name1.rstrip(".fasta")
                        for name in glob.glob(name1+"_part*.fasta"):
                            encoder.read(name)
                            encoder.encode()
                            num = name[name.find("part") + 4:-6]
                            encoder.dump(name1+"_part"+num)
                else:
                    devide(seqPath, chopSize, outPrefix)
                    for name in glob.glob(outPrefix+"_part*.fasta"):
                        encoder.read(name)
                        encoder.encode()
                        num = name[name.find("part") + 4:-6]
                        encoder.dump(outPrefix+"_part"+num)

        elif not (GOfile is None):
            if outPrefix == GOfile:
                outPrefix = GOfile[:-4]
            if not GOpartioned:
                oneHotencd = GOencoder()
                if Protfilter != "":
                    oneHotencd.load_filter(Protfilter, GOfilter)
                oneHotencd.read(GOfile)
                oneHotencd.encode()
                oneHotencd.dump(outPrefix)
            else:
                for name in glob.glob(Protfilter+"_part*_keys.txt"):
                    oneHotencd = GOencoder()
                    oneHotencd.load_filter(name, GOfilter)
                    oneHotencd.read(GOfile)
                    oneHotencd.encode()
                    num = name[name.find("part") + 4:-9]
                    oneHotencd.dump(outPrefix+"_part"+num)

        elif not (predictionFile is None):
            if outPrefix == predictionFile:
                suffix = "predictions.txt"
                if "/" in outPrefix:
                    outPrefix = outPrefix[:outPrefix.rfind("/")]+"/"+suffix
                else:
                    outPrefix = suffix
            oneHotencd = GOencoder()
            oneHotencd.decode(predictionFile, GOfilter, outPrefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
