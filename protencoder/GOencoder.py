from protencoder.protencoder import encoder
import glob
import numpy as np

class GOencoder(encoder):
    def __init__(self):
        self.handler = encoder()
        self.GOclasses = {'F': set(), 'P': set(), 'C': set()}

    def encode(self):
        if self.handler.GOfilter['F'] != []:
            self.GOclasses = self.handler.GOfilter
        else:
            for prot in self.handler.seqDict:
                for GOclass in self.handler.seqDict[prot]:
                    for GOA in self.handler.seqDict[prot][GOclass]:
                        self.GOclasses[GOclass].add(GOA)
        for GOclass in self.GOclasses:
            self.GOclasses[GOclass] = list(self.GOclasses[GOclass])
            self.GOclasses[GOclass] = {b: a for a,
                                       b in enumerate(self.GOclasses[GOclass])}
        for seq in self.handler.seqDict:
            for GOclass in self.handler.seqDict[seq]:
                encoded = len(self.GOclasses[GOclass]) * [0]
                for GOA in self.handler.seqDict[seq][GOclass]:
                    encoded[self.GOclasses[GOclass][GOA]] = 1
                self.handler.seqDict[seq][GOclass] = encoded

    def decode(self, npyPrefix, keyPrefix, outPrefix):
        keysFile = open(glob.glob(keyPrefix)[0])
        keysList = []
        for line in keysFile.readlines():
            line = line.rstrip('\n').split('\t')
            if line[1] == 'F':
                keysList.append(line[0])
            else:
                break
        keysFile.close()
        report = open(outPrefix, 'w')
        report.write("AUTHOR\tGOlite\nMODEL\t1\n")
        report.write("KEYWORDS\tmachine learning.\n")
        for file in glob.glob(npyPrefix):
            prediction = np.load(file)
            predictionKeys = file[:file.rfind("_")]
            predictionKeys = predictionKeys[:predictionKeys.rfind("_")] + "_keys.txt"
            prdctKeysFile = open(predictionKeys)
            prdctKeysList = []
            for line in prdctKeysFile.readlines():
                prdctKeysList.append(line.rstrip('\n'))
            prdctKeysFile.close()
            for prot in range(prediction.shape[0]):
                for GO in range(len(prediction[prot])):
                    if prediction[prot][GO] == 1:
                        report.write(prdctKeysList[prot]+'\t'+keysList[GO]+'\n')
        report.close()

    def read(self, GOfile):
        self.handler.read_GO(GOfile)

    def dump(self, outPrefix):
        self.handler.dump_GO(self.GOclasses, outPrefix)

    def load_filter(self, Protfilter, GOfilter):
        self.handler.load_filter(Protfilter)
        self.handler.load_GO_filter(GOfilter)
