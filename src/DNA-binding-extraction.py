import csv


def peak_extraction():
    with open('../res/DNA-binding/all_binding_event.tsv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        next(reader)
        peak_center = []
        for line in reader:
            if line[-1] == 'IND' or line[-1] == 'REP':
                peak_center.append(line[2])

    return peak_center

def sequence_extraction(peak_center):
    with open('../res/DNA-binding/peak-sequence.tsv.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        next(reader)
        for line in reader:
            center = line[7]
            start = int(line[5])
            stop = int(line[6])
            sequence = line[8]
            sequence =sequence.strip()
            if '...' in sequence:
                sequence = sequence.split('...')[0]
            if center in peak_center:
                center = int(center)
                upstream = center - start
                downstream =stop - center
                total = stop - start
                DNA = sequence
                if len(sequence) < 101:
                    pass
                if len(sequence) >= 101:
                    difference = len(sequence) - 101
                    if difference %2 ==0:
                        DNA = sequence[difference/2: len(sequence)-(difference/2)]
                        #print sequence, DNA
                    else:
                        DNA = sequence[difference/2: len(sequence) - (difference/2)-1]
                    print DNA
                    
                


def main():
    peak_center = peak_extraction()
    sequence_extraction(peak_center)
    
if '__name__' == '__main__':
    main()

    
