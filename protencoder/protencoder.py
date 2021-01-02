from Bio import SeqIO
import numpy as np


class encoder(object):
    def __init__(self):
        self.seqDict = dict()
        self.filter = list()
        self.GOfilter = {'F': list(), 'P': list(), 'C': list()}
        self.seqKeys = list()

    def read_fasta(self, seqPath):
        fasta_sequences = SeqIO.parse(seqPath, 'fasta')
        for fasta in fasta_sequences:
            if fasta.id in self.filter or self.filter == []:
                seq = str(fasta.seq)
                if "*" in seq:
                    seq = seq.replace("*", "")
                self.seqDict[fasta.id] = seq

    def read_GO(self, filePath):
        if self.filter != []:
            self.seqKeys = self.filter
        keysF = open(filePath)
        for line in keysF.readlines():
            line = line.rstrip("\n").split("\t")
            if self.GOfilter['F'] != []:
                if (line[0] in self.filter or self.filter == []):
                    if line[0] not in self.seqDict:
                        self.seqDict[line[0]] = {'F': [], 'P': [], 'C': []}
                    if line[1] in self.GOfilter[line[2]]:
                        self.seqDict[line[0]][line[2]].append(line[1])
                        if self.filter == []:
                            self.seqKeys.append(line[0])
        keysF.close()
        self.seqKeys = [x for x in self.seqKeys if x in self.seqDict]

    def dump(self, outPrefix, method):
        seqArr = np.array(list(self.seqDict.values()))
        if self.filter == []:
            self.seqKeys = list(self.seqDict.keys())
        else:
            self.seqKeys = self.filter
        self.seqDict = dict()
        np.save(outPrefix + "_" + method, seqArr)
        del seqArr
        seqKeysF = open(outPrefix + "_keys.txt", 'w')
        for i in self.seqKeys:
            seqKeysF.write(i + "\n")
        seqKeysF.close()

    def dump_GO(self, GOclasses, outPrefix):
        FOrderedGOA = []
        POrderedGOA = []
        COrderedGOA = []
        for seq in self.seqKeys:
            FOrderedGOA.append(self.seqDict[seq]['F'])
            POrderedGOA.append(self.seqDict[seq]['P'])
            COrderedGOA.append(self.seqDict[seq]['C'])
            self.seqDict.pop(seq)
        FOrderedGOA = np.array(FOrderedGOA)
        POrderedGOA = np.array(POrderedGOA)
        COrderedGOA = np.array(COrderedGOA)
        np.save(outPrefix + "_FGOA", FOrderedGOA)
        del FOrderedGOA
        np.save(outPrefix + "_PGOA", POrderedGOA)
        del POrderedGOA
        np.save(outPrefix + "_CGOA", COrderedGOA)
        del COrderedGOA
        FGOAf = open(outPrefix + "_FGOclasses.txt", 'w')
        PGOAf = open(outPrefix + "_PGOclasses.txt", 'w')
        CGOAf = open(outPrefix + "_CGOclasses.txt", 'w')
        for GOA in GOclasses['F']:
            FGOAf.write(GOA + "\n")
        for GOA in GOclasses['P']:
            PGOAf.write(GOA + "\n")
        for GOA in GOclasses['C']:
            CGOAf.write(GOA + "\n")
        FGOAf.close()
        PGOAf.close()
        CGOAf.close()

    def load_filter(self, filter):
        seqKeysF = open(filter)
        for i in seqKeysF.readlines():
            self.filter.append(i.rstrip("\n"))
        seqKeysF.close()

    def load_GO_filter(self, filter):
        GOKeysF = open(filter)
        for i in GOKeysF.readlines():
            i = i.rstrip("\n").split('\t')
            self.GOfilter[i[1]].append(i[0])
        GOKeysF.close()
