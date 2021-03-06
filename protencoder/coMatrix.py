from protencoder.protencoder import encoder
from cv2 import resize, INTER_LINEAR, INTER_AREA
import numpy as np
import random


class AAcomptability():
    def __init__(self, dsize=(500, 500), action='pad'):
        self.handler = encoder()
        self.dsize = (dsize, dsize)
        self.action = action
        self.SCM, self.HCM, self.CCM = get_data()
        self.matrices = [self.SCM, self.HCM, self.CCM]

    def encode(self):
        # self.aaDict['X'] = self.Nprops * [0.5]
        # self.aa['X'] = list(range(len(self.aaDict)-1))
        # self.aaDict['U'] = self.aaDict['C']
        # self.aa['U'] = self.aa['C']
        # self.aaDict['O'] = self.aaDict['K']
        # self.aa['O'] = self.aa['K']
        # # adding 'B' for 'N'/'D', and 'Z' for 'Q'/'E', and 'J' for 'L'/'I'
        # summ = [sum(i) for i in zip(self.aaDict['N'], self.aaDict['D'])]
        # self.aaDict['B'] = [x/2 for x in summ]
        # self.aa['B'] = self.aa['N'] + self.aa['D']
        # summ = [sum(i) for i in zip(self.aaDict['Q'], self.aaDict['E'])]
        # self.aaDict['Z'] = [x/2 for x in summ]
        # self.aa['Z'] = self.aa['Q'] + self.aa['E']
        # summ = [sum(i) for i in zip(self.aaDict['L'], self.aaDict['I'])]
        # self.aaDict['J'] = [x/2 for x in summ]
        # self.aa['J'] = self.aa['L'] + self.aa['I']
        for prot in self.handler.seqDict:
            seq = self.handler.seqDict[prot][:1000]
            encoded = np.zeros((len(self.matrices), len(seq), len(seq)),
                               dtype='uint8')
            Xs = ['R', 'K', 'D', 'Q', 'N', 'E', 'H', 'S', 'T',
                  'P', 'Y', 'C', 'G', 'A', 'M', 'W', 'L', 'V',
                  'F', 'I']
            Bs = ['N', 'D']
            Zs = ['Q', 'E']
            Js = ['L', 'I']
            while ('X' in seq):
                aa = Xs[random.randint(0, len(Xs)-1)]
                seq = seq.replace("X", aa, 1)
            while ('B' in seq):
                aa = Bs[random.randint(0, len(Bs)-1)]
                seq = seq.replace("B", aa, 1)
            while ('Z' in seq):
                aa = Zs[random.randint(0, len(Zs)-1)]
                seq = seq.replace("Z", aa, 1)
            while ('J' in seq):
                aa = Js[random.randint(0, len(Js)-1)]
                seq = seq.replace("J", aa, 1)
            if 'U' in seq:
                seq = seq.replace("U", 'C')
            if 'O' in seq:
                seq = seq.replace("O", "K")
            for i in range(len(self.matrices)):
                for j in range(len(seq)):
                    for k in range(len(seq)):
                        A = seq[j]
                        B = seq[k]
                        if k <= j:
                            score = (self.matrices[i][A][B]) * 255/20
                            encoded[i][j][k] = score
                            encoded[i][k][j] = score
                        else:
                            break
            encoded = self.co_resize(encoded)
            self.handler.seqDict[prot] = encoded.astype('uint8')

    def co_resize(self, prot):
        if prot.shape[1] > self.dsize[0]:
            x = np.moveaxis(prot, 0, -1)
            x = resize(x, self.dsize, interpolation=INTER_AREA)
            x = np.moveaxis(x, -1, 0)
        elif prot.shape[1] < self.dsize[0]:
            if self.action == "repeat":
                repeatSize = int(self.dsize[0]/prot.shape[1])
                x = np.repeat(prot, repeatSize, axis=1)
                x = np.repeat(x, repeatSize, axis=2)
                padSize = self.dsize[0] - x.shape[1]
                x = np.pad(x, ((0, 0), (0, padSize), (0, padSize)),
                           mode="constant")
            elif self.action == "tile":
                tileSize = int(self.dsize[0]/prot.shape[1])+1
                x = np.tile(prot, (1, tileSize, tileSize))
                x = x[:, :self.dsize[0], :self.dsize[0]]
            elif self.action == "resize":
                x = np.moveaxis(prot, 0, -1)
                x = resize(x, self.dsize, interpolation=INTER_LINEAR)
                x = np.moveaxis(x, -1, 0)
            elif self.action == "pad":
                padSize = self.dsize[0] - prot.shape[1]
                x = np.pad(prot, ((0, 0), (0, padSize), (0, padSize)),
                           mode="constant")
        else:
            x = prot
        return np.moveaxis(x, 0, -1)

    def read(self, seqPath):
        self.handler.read_fasta(seqPath)

    def dump(self, outPrefix):
        self.handler.dump(outPrefix, "_comatrix")


def get_data():
    mw = [71, 103, 115, 129, 147, 57, 137, 113, 128, 113, 131, 114,
          97, 128, 156, 87, 101, 99, 186, 163]
    mw = [a-56 for a in mw]
    aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
          'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    indxs = {b: a for a, b in enumerate(aa)}
    SCM = {a: {b: 0 for b in aa} for a in aa}

    for a in SCM:
        for b in SCM[a]:
            a_mw = mw[indxs[a]]
            b_mw = mw[indxs[b]]
            SCI = 20-abs((a_mw+b_mw-123) * (19/135))
            SCM[a][b] = int(float(SCI*100))/100

    HCM = {'R': {'R': 20.0, 'K': 14.8, 'D': 11.9, 'Q': 11.8, 'N': 11.4, 'E': 11.2,
             'H': 9.6, 'S': 8.5, 'T': 7.9, 'P': 7.1, 'Y': 6.4, 'C': 6.3, 'G': 5.4,
             'A': 4.8, 'M': 4.6, 'W': 3.9, 'L': 2.6, 'V': 2.4, 'F': 2.1, 'I': 1.0},
             'K': {'R': 14.8, 'K': 20.0, 'D': 17.1, 'Q': 17.0, 'N': 16.6, 'E': 16.4,
             'H': 14.8, 'S': 13.7, 'T': 13.1, 'P': 12.3, 'Y': 11.6, 'C': 11.5, 'G': 10.6,
             'A': 10.0, 'M': 9.8, 'W': 9.1, 'L': 7.8, 'V': 7.6, 'F': 7.3, 'I': 6.2},
             'D': {'R': 11.9, 'K': 17.1, 'D': 20.0, 'Q': 19.8, 'N': 19.5, 'E': 19.3,
             'H': 17.7, 'S': 16.6, 'T': 16.0, 'P': 15.1, 'Y': 14.5, 'C': 14.3, 'G': 13.4,
             'A': 12.8, 'M': 12.7, 'W': 11.9, 'L': 10.7, 'V': 10.5, 'F': 10.1, 'I': 9.1},
             'Q': {'R': 11.8, 'K': 17.0, 'D': 19.8, 'Q': 20.0, 'N': 19.6, 'E': 19.6,
             'H': 17.8, 'S': 16.8, 'T': 16.1, 'P': 16.3, 'Y': 14.3, 'C': 14.6, 'G': 13.6,
             'A': 13.0, 'M': 12.8, 'W': 12.1, 'L': 10.9, 'V': 10.7, 'F': 10.3, 'I': 9.2},
             'N': {'R': 11.4, 'K': 16.6, 'D': 19.6, 'Q': 19.6, 'N': 20.0, 'E': 19.8,
             'H': 18.2, 'S': 17.1, 'T': 16.5, 'P': 15.7, 'Y': 16.0, 'C': 14.9, 'G': 14.0,
             'A': 13.4, 'M': 13.2, 'W': 12.6, 'L': 11.2, 'V': 11.0, 'F': 10.7, 'I': 9.6},
             'E': {'R': 11.2, 'K': 16.4, 'D': 19.3, 'Q': 19.5, 'N': 19.8, 'E': 20.0,
             'H': 18.4, 'S': 17.3, 'T': 16.7, 'P': 15.9, 'Y': 15.2, 'C': 15.0, 'G': 14.1,
             'A': 13.5, 'M': 13.4, 'W': 12.7, 'L': 11.4, 'V': 11.2, 'F': 10.9, 'I': 9.8},
             'H': {'R': 9.6, 'K': 14.2, 'D': 17.7, 'Q': 17.8, 'N': 18.2, 'E': 18.4,
             'H': 20.0, 'S': 18.9, 'T': 18.3, 'P': 17.5, 'Y': 16.2, 'C': 16.6, 'G': 15.8,
             'A': 15.2, 'M': 15.0, 'W': 14.3, 'L': 13.0, 'V': 12.8, 'F': 12.5, 'I': 11.4},
             'S': {'R': 8.5, 'K': 13.7, 'D': 16.6, 'Q': 16.8, 'N': 17.1, 'E': 17.3,
             'H': 18.9, 'S': 20.0, 'T': 19.4, 'P': 18.5, 'Y': 17.9, 'C': 17.7, 'G': 16.8,
             'A': 16.2, 'M': 16.1, 'W': 15.3, 'L': 14.1, 'V': 13.9, 'F': 13.5, 'I': 12.5},
             'T': {'R': 7.9, 'K': 13.1, 'D': 16.0, 'Q': 16.1, 'N': 16.5, 'E': 16.7,
             'H': 18.3, 'S': 19.4, 'T': 20.0, 'P': 19.2, 'Y': 18.5, 'C': 18.4, 'G': 17.5,
             'A': 16.9, 'M': 16.7, 'W': 16.0, 'L': 14.7, 'V': 14.5, 'F': 14.2, 'I': 13.1},
             'P': {'R': 7.1, 'K': 12.3, 'D': 16.1, 'Q': 15.3, 'N': 16.7, 'E': 16.9,
             'H': 17.6, 'S': 18.6, 'T': 19.2, 'P': 20.0, 'Y': 19.3, 'C': 19.2, 'G': 18.3,
             'A': 17.7, 'M': 17.6, 'W': 16.8, 'L': 16.6, 'V': 16.4, 'F': 16.0, 'I': 13.9},
             'Y': {'R': 6.4, 'K': 11.6, 'D': 14.6, 'Q': 14.7, 'N': 16.0, 'E': 16.2,
             'H': 16.8, 'S': 17.9, 'T': 18.5, 'P': 19.3, 'Y': 20.0, 'C': 19.8, 'G': 18.9,
             'A': 18.4, 'M': 18.2, 'W': 17.6, 'L': 16.2, 'V': 16.0, 'F': 15.7, 'I': 14.6},
             'C': {'R': 6.3, 'K': 11.5, 'D': 14.3, 'Q': 14.5, 'N': 14.9, 'E': 15.0,
             'H': 16.6, 'S': 17.7, 'T': 18.4, 'P': 19.2, 'Y': 19.8, 'C': 20.0, 'G': 19.1,
             'A': 18.5, 'M': 18.3, 'W': 17.6, 'L': 16.4, 'V': 16.2, 'F': 15.8, 'I': 14.7},
             'G': {'R': 5.4, 'K': 10.6, 'D': 13.4, 'Q': 13.6, 'N': 14.0, 'E': 14.1,
             'H': 15.8, 'S': 16.8, 'T': 17.5, 'P': 18.3, 'Y': 18.9, 'C': 19.1, 'G': 20.0,
             'A': 19.4, 'M': 19.2, 'W': 18.5, 'L': 17.3, 'V': 17.1, 'F': 16.7, 'I': 15.6},
             'A': {'R': 4.8, 'K': 10.0, 'D': 12.8, 'Q': 13.0, 'N': 13.4, 'E': 13.5,
             'H': 15.2, 'S': 16.2, 'T': 16.9, 'P': 17.7, 'Y': 18.4, 'C': 18.5, 'G': 19.4,
             'A': 20.0, 'M': 19.8, 'W': 19.1, 'L': 17.8, 'V': 17.7, 'F': 17.3, 'I': 16.2},
             'M': {'R': 4.6, 'K': 9.8, 'D': 12.7, 'Q': 12.8, 'N': 13.2, 'E': 13.4,
             'H': 15.0, 'S': 16.1, 'T': 16.7, 'P': 17.5, 'Y': 18.2, 'C': 18.3, 'G': 19.2,
             'A': 19.8, 'M': 20.0, 'W': 19.3, 'L': 18.0, 'V': 17.8, 'F': 17.5, 'I': 16.4},
             'W': {'R': 3.9, 'K': 9.1, 'D': 11.9, 'Q': 12.1, 'N': 12.5, 'E': 12.7,
             'H': 14.3, 'S': 16.3, 'T': 16.0, 'P': 16.8, 'Y': 17.5, 'C': 17.6, 'G': 18.5,
             'A': 19.1, 'M': 19.3, 'W': 20.0, 'L': 18.7, 'V': 18.6, 'F': 18.2, 'I': 17.1},
             'L': {'R': 2.6, 'K': 7.8, 'D': 10.7, 'Q': 10.9, 'N': 11.2, 'E': 11.4,
             'H': 13.0, 'S': 14.1, 'T': 14.7, 'P': 15.6, 'Y': 16.2, 'C': 16.4, 'G': 17.3,
             'A': 17.8, 'M': 18.0, 'W': 18.7, 'L': 20.0, 'V': 19.8, 'F': 19.6, 'I': 18.4},
             'V': {'R': 2.4, 'K': 7.6, 'D': 10.5, 'Q': 10.7, 'N': 11.0, 'E': 11.2,
             'H': 12.9, 'S': 13.9, 'T': 14.5, 'P': 15.4, 'Y': 16.0, 'C': 16.2, 'G': 17.1,
             'A': 17.7, 'M': 17.9, 'W': 19.6, 'L': 19.8, 'V': 20.0, 'F': 19.6, 'I': 18.6},
             'F': {'R': 2.1, 'K': 7.3, 'D': 10.1, 'Q': 10.3, 'N': 10.7, 'E': 10.9,
             'H': 12.5, 'S': 13.5, 'T': 14.2, 'P': 15.0, 'Y': 15.7, 'C': 15.8, 'G': 16.7,
             'A': 17.3, 'M': 17.5, 'W': 18.2, 'L': 19.5, 'V': 19.6, 'F': 20.0, 'I': 18.9},
             'I': {'R': 1.0, 'K': 6.2, 'D': 9.1, 'Q': 9.2, 'N': 9.6, 'E': 9.8, 'H': 11.4,
             'S': 12.5, 'T': 13.1, 'P': 13.9, 'Y': 14.6, 'C': 14.7, 'G': 15.6, 'A': 16.2,
             'M': 16.4, 'W': 17.1, 'L': 18.4, 'V': 18.6, 'F': 18.9, 'I': 20.0}}

    CCM = {'D': {'D': 0.9, 'E': 2.0, 'C': 6.4, 'N': 7.2, 'F': 7.4, 'Q': 7.8,
           'Y': 7.8, 'S': 7.9, 'M': 8.0, 'T': 8.3, 'I': 8.4, 'G': 8.6, 'V': 8.6, 'W': 8.6,
           'L': 8.6, 'A': 8.6, 'P': 9.8, 'H': 12.4, 'K': 17.5, 'R': 19.9},
           'E': {'D': 2.0, 'E': 3.0, 'C': 6.9, 'N': 7.6, 'F': 7.8, 'Q': 8.1, 'Y': 8.2,
           'S': 8.2, 'M': 8.3, 'T': 8.6, 'I': 8.7, 'G': 8.8, 'V': 8.8, 'W': 8.8,
           'L': 8.8, 'A': 8.9, 'P': 9.9, 'H': 12.3, 'K': 16.8, 'R': 19.0},
           'C': {'D': 6.4, 'E': 6.9, 'C': 8.9, 'N': 9.3, 'F': 9.4, 'Q': 9.5, 'Y': 9.5,
           'S': 9.6, 'M': 9.6, 'T': 9.8, 'I': 9.8, 'G': 9.9, 'V': 9.9, 'W': 9.9,
           'L': 9.9, 'A': 9.9, 'P': 10.4, 'H': 11.6, 'K': 14.0, 'R': 15.1},
           'N': {'D': 7.2, 'E': 7.6, 'C': 9.3, 'N': 9.6, 'F': 9.6, 'Q': 9.8, 'Y': 9.8,
           'S': 9.8, 'M': 9.9, 'T': 10.0, 'I': 10.0, 'G': 10.1, 'V': 10.1, 'W': 10.1,
           'L': 10.1, 'A': 10.1, 'P': 10.5, 'H': 11.5, 'K': 13.4, 'R': 14.4},
           'F': {'D': 7.4, 'E': 7.8, 'C': 9.4, 'N': 9.6, 'F': 9.7, 'Q': 9.8, 'Y': 9.9,
           'S': 9.9, 'M': 9.9, 'T': 10.0, 'I': 10.1, 'G': 10.1, 'V': 10.1, 'W': 10.1,
           'L': 10.1, 'A': 10.2, 'P': 10.6, 'H': 11.5, 'K': 13.3, 'R': 14.2},
           'Q': {'D': 7.8, 'E': 8.1, 'C': 9.5, 'N': 9.8, 'F': 9.8, 'Q': 10.0, 'Y': 10.0,
           'S': 10.0, 'M': 10.0, 'T': 10.1, 'I': 10.2, 'G': 10.2, 'V': 10.2, 'W': 10.2,
           'L': 10.2, 'A': 10.2, 'P': 10.6, 'H': 11.4, 'K': 13.1, 'R': 13.9},
           'Y': {'D': 7.8, 'E': 8.2, 'C': 9.5, 'N': 9.8, 'F': 9.9, 'Q': 10.0, 'Y': 10.0,
           'S': 10.0, 'M': 10.1, 'T': 10.1, 'I': 10.2, 'G': 10.2, 'V': 10.2, 'W': 10.2,
           'L': 10.2, 'A': 10.3, 'P': 10.6, 'H': 11.4, 'K': 13.1, 'R': 13.8},
           'S': {'D': 7.9, 'E': 8.2, 'C': 9.6, 'N': 9.8, 'F': 9.9, 'Q': 10.0, 'Y': 10.0,
           'S': 10.0, 'M': 10.1, 'T': 10.2, 'I': 10.2, 'G': 10.2, 'V': 10.2, 'W': 10.2,
           'L': 10.2, 'A': 10.3, 'P': 10.6, 'H': 11.4, 'K': 13.0, 'R': 13.8},
           'M': {'D': 8.0, 'E': 8.3, 'C': 9.6, 'N': 9.9, 'F': 9.9, 'Q': 10.0, 'Y': 10.1,
           'S': 10.1, 'M': 10.1, 'T': 10.2, 'I': 10.2, 'G': 10.3, 'V': 10.3, 'W': 10.3,
           'L': 10.3, 'A': 10.3, 'P': 10.6, 'H': 11.4, 'K': 12.9, 'R': 13.7},
           'T': {'D': 8.3, 'E': 8.6, 'C': 9.8, 'N': 10.0, 'F': 10.0, 'Q': 10.1,
           'Y': 10.1, 'S': 10.2, 'M': 10.2, 'T': 10.3, 'I': 10.3, 'G': 10.3, 'V': 10.3,
           'W': 10.4, 'L': 10.4, 'A': 10.4, 'P': 10.7, 'H': 11.4, 'K': 12.7, 'R': 13.4},
           'I': {'D': 8.4, 'E': 8.7, 'C': 9.8, 'N': 10.0, 'F': 10.1, 'Q': 10.2,
           'Y': 10.2, 'S': 10.2, 'M': 10.2, 'T': 10.3, 'I': 10.3, 'G': 10.4, 'V': 10.4,
           'W': 10.4, 'L': 10.4, 'A': 10.4, 'P': 10.7, 'H': 11.4, 'K': 12.7, 'R': 13.3},
           'G': {'D': 8.6, 'E': 8.8, 'C': 9.9, 'N': 10.1, 'F': 10.1, 'Q': 10.2,
           'Y': 10.2, 'S': 10.2, 'M': 10.3, 'T': 10.3, 'I': 10.4, 'G': 10.4, 'V': 10.4,
           'W': 10.4, 'L': 10.4, 'A': 10.4, 'P': 10.7, 'H': 11.3, 'K': 12.6, 'R': 13.2},
           'V': {'D': 8.6, 'E': 8.8, 'C': 9.9, 'N': 10.1, 'F': 10.1, 'Q': 10.2,
           'Y': 10.2, 'S': 10.2, 'M': 10.3, 'T': 10.3, 'I': 10.4, 'G': 10.4, 'V': 10.4,
           'W': 10.4, 'L': 10.4, 'A': 10.4, 'P': 10.7, 'H': 11.3, 'K': 12.6, 'R': 13.2},
           'W': {'D': 8.6, 'E': 8.8, 'C': 9.9, 'N': 10.1, 'F': 10.1, 'Q': 10.2,
           'Y': 10.2, 'S': 10.2, 'M': 10.3, 'T': 10.4, 'I': 10.4, 'G': 10.4, 'V': 10.4,
           'W': 10.4, 'L': 10.4, 'A': 10.4, 'P': 10.7, 'H': 11.3, 'K': 12.6, 'R': 13.2},
           'L': {'D': 8.6, 'E': 8.8, 'C': 9.9, 'N': 10.1, 'F': 10.1, 'Q': 10.2,
           'Y': 10.2, 'S': 10.2, 'M': 10.3, 'T': 10.4, 'I': 10.4, 'G': 10.4, 'V': 10.4,
           'W': 10.4, 'L': 10.4, 'A': 10.4, 'P': 10.7, 'H': 11.3, 'K': 12.6, 'R': 13.2},
           'A': {'D': 8.6, 'E': 8.9, 'C': 9.9, 'N': 10.1, 'F': 10.2, 'Q': 10.2,
           'Y': 10.3, 'S': 10.3, 'M': 10.3, 'T': 10.4, 'I': 10.4, 'G': 10.4, 'V': 10.4,
           'W': 10.4, 'L': 10.4, 'A': 10.4, 'P': 10.7, 'H': 11.3, 'K': 12.5, 'R': 13.1},
           'P': {'D': 9.8, 'E': 9.9, 'C': 10.4, 'N': 10.5, 'F': 10.6, 'Q': 10.6,
           'Y': 10.6, 'S': 10.6, 'M': 10.6, 'T': 10.7, 'I': 10.7, 'G': 10.7, 'V': 10.7,
           'W': 10.7, 'L': 10.7, 'A': 10.7, 'P': 10.8, 'H': 11.2, 'K': 11.8, 'R': 12.1},
           'H': {'D': 12.4, 'E': 12.3, 'C': 11.6, 'N': 11.5, 'F': 11.5, 'Q': 11.4,
           'Y': 11.4, 'S': 11.4, 'M': 11.4, 'T': 11.4, 'I': 11.4, 'G': 11.3, 'V': 11.3,
           'W': 11.3, 'L': 11.3, 'A': 11.3, 'P': 11.2, 'H': 10.8, 'K': 10.1, 'R': 9.8},
           'K': {'D': 17.5, 'E': 16.8, 'C': 14.0, 'N': 13.4, 'F': 13.3, 'Q': 13.1,
           'Y': 13.1, 'S': 13.0, 'M': 12.9, 'T': 12.7, 'I': 12.7, 'G': 12.6, 'V': 12.6,
           'W': 12.6, 'L': 12.6, 'A': 12.5, 'P': 11.8, 'H': 10.1, 'K': 6.8, 'R': 5.2},
           'R': {'D': 19.9, 'E': 19.0, 'C': 15.1, 'N': 14.4, 'F': 14.2, 'Q': 13.9,
           'Y': 13.8, 'S': 13.8, 'M': 13.7, 'T': 13.4, 'I': 13.3, 'G': 13.2, 'V': 13.2,
           'W': 13.2, 'L': 13.2, 'A': 13.1, 'P': 12.1, 'H': 9.8, 'K': 5.2, 'R': 3.1}}
    return SCM, HCM, CCM
