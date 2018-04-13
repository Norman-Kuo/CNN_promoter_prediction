## To Install keras, go to https://keras.io/#installation to install tensorflow first, then keras

import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Flatten, Dropout
from keras.activations import relu
from keras.layers.pooling import MaxPooling1D
from keras.layers import LSTM
from keras.optimizers import SGD
import random
import sys

promoter = '../res/experimental_promoter_sequence.tsv'
non_promoter = '../res/non_promoter_sequence.tsv'


class Promoter_Prediction:
    def __init__(self, percentage = 90):
        self.percentage = percentage

    def Sequence_extraction(self):
        positive_set = [] # positive list
        negative_set = [] # negative list
        Train_pos = [] # positive Training sequences list
        Test_pos = [] # positive Testing sequences list
        Train_neg = [] # negative Training sequences list
        Test_neg = [] # negative Testing sequences list
        with open(promoter, 'U') as csvfile: #open promoter data
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL) #csv reader tab separate
            for line in reader:
                positive_set.append(line) # add all the line to the positive set         
        
        num_sample = len(positive_set) #calculate the # of sequence so we can use same for negative set

        seq_length = len(positive_set[1][1]) #calculate the sequence length

        percentage = float(self.percentage)/100 #The input percentage will determine your percentage of training sample number from the available provided sample (Usually positive sample)

        Train_sample_number = int(round(num_sample*percentage, 0)) #Determine training sample number from the percentage provided
        Test_sample_number = num_sample - Train_sample_number #minus training sample number from total sample number

        Train_pos_set =random.sample(positive_set, Train_sample_number) #Create a randomize positive sample for training.


        #This creates a testing positive set, but then remove all the sequences from the training set so we don't have same sequences in training and testing
        Test_pos_set = positive_set 
        for i in Train_pos_set:
            if i in Test_pos_set:
                Test_pos_set.remove(i)

        #Extract sequences from the training positive set
        for sequences in Train_pos_set:
            sequence = [] #refresh sequence for every loop
            for nucleotide in sequences[1]:
                sequence.append(nucleotide) #add nucleotide to the sequences
            Train_pos.append(sequence) #add the sequence to the Train_positive set

        #Extract sequence from the testing positive set
        for sequences in Test_pos_set:
            sequence = [] #refresh sequence for every loop
            for nucleotide in sequences[1]:
                sequence.append(nucleotide) #add nucleotide to the sequences
            Test_pos.append(sequence) #add the sequence to the Train_positive set

        with open(non_promoter, 'U') as csvfile: #open non promoter data
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)#csv reader tab separate
            for line in reader:
                negative_set.append(line) # add all the line to the negative set

        neg = random.sample(negative_set, num_sample) #Create a negative set data using the same number of sample as positive set
        Train_neg_set = random.sample(neg, Train_sample_number) #Create a randomize negative sample for training.

        #This creates a testing negative set, but then remove all the sequences from the training set so we don't have same sequences in training and testing
        Test_neg_set = neg
        for i in Train_neg_set:
            if i in Test_neg_set:
                Test_neg_set.remove(i)

        #Extract sequence from training negative set
        for sequences in Train_neg_set:
            sequence = [] #refresh sequence for every loop
            for nucleotide in sequences[1]:
                sequence.append(nucleotide) #add nucleotide to the sequences
            Train_neg.append(sequence) #add the sequence to the Train_positive set

        #Extract sequence from testinging negative set
        for sequences in Test_neg_set:
            sequence = [] #refresh sequence for every loop
            for nucleotide in sequences[1]:
                sequence.append(nucleotide) #add nucleotide to the sequences
            Test_neg.append(sequence) #add the sequence to the Train_positive set
        

        
        # Make the training and testing positive set into array using numpy        
        Train_pos_array = np.array(Train_pos)
        Test_pos_array = np.array(Test_pos)

        # Make the training and testing negative set into array using numpy   
        Train_neg_array = np.array(Train_neg)
        Test_neg_array = np.array(Test_neg)


        #Using numpy we create zeros matrices base on number of nucleotide in list (now set as 60 nucleotide), then expand it times 4 since there are 4 possible nucleotides (ATGC) available at each position
        pos_tensor = np.zeros(list(Train_pos_array.shape) + [4])
        neg_tensor = np.zeros(list(Train_neg_array.shape) + [4])
        Test_pos_tensor = np.zeros(list(Test_pos_array.shape) + [4])
        Test_neg_tensor = np.zeros(list(Test_neg_array.shape) + [4])


        
        #This creates a dictionary for if nucleotides read, we assign it to the position.
        # A = [1, 0, 0, 0]
        # C = [0, 1, 0, 0]
        # G = [0, 0, 1, 0]
        # T = [0, 0, 0, 1]
        base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        #Naive one-hot encoding
        for row in range(Train_sample_number): # Counting each sample in the training sample set
            for col in range(seq_length): # For each sample, from the sequence length (60) start from position 1 (or 0 in python)
                #This creates lists that has same number of training sample number, within each list contains list that has same number of length of sequence, and within those list contain the position of each nucleotide based off dictionary
                pos_tensor[row,col,base_dict[Train_pos_array[row,col]]] = 1 
                neg_tensor[row,col,base_dict[Train_neg_array[row,col]]] = 1
        
        for row in range(Test_sample_number): # Counting each sample in the testing sample set
            for col in range(seq_length): # For each sample, from the sequence length (60) start from position 1 (or 0 in python)
                #This creates lists that has same number of testing sample number, within each list contains list that has same number of length of sequence, and within those list contain the position of each nucleotide based off dictionary
                Test_pos_tensor[row,col,base_dict[Test_pos_array[row,col]]] = 1
                Test_neg_tensor[row,col,base_dict[Test_neg_array[row,col]]] = 1

        
            
        print('Positive sample matrix shape: {}'.format(Train_pos_array.shape)) #Shows the matrix shape of original training set contain number sample and sequence length
        # this should be a 3D tensor with shape: (samples, steps, input_dim)
        print('Positive sample tensor shape: {}'.format(pos_tensor.shape)) #shows the matrix shape of tensor that contains numbers of samples and sequences length and nucleotides (4)
        
        #X stack the list of positive sequences follow by list negative sequences
        X = np.vstack((pos_tensor, neg_tensor))
        #y stack the positive response (1) follow by negative response (0) in the list
        y = np.concatenate((np.ones(Train_sample_number), np.zeros(Train_sample_number)))


        return X, y, seq_length, pos_tensor, neg_tensor, Test_pos_tensor, Test_neg_tensor

    def sequence_extraction_real(self, complement_list):
        #complement_list = fasta_sequence(fasta)
        #Make the input sequence into array using numpy
        complement_array = np.array(complement_list)
        complement_tensor = np.zeros(list(complement_array.shape) + [4])
        base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        for row in range(len(complement_list)):
            for col in range(101):
                complement_tensor[row,col,base_dict[complement_array[row,col]]] = 1
        return complement_tensor


    def fasta_sequence(self, fasta):
        with open (fasta, 'U') as filehandler:
            name = str(fasta.split('/')[-1])
            #print(name)
            complement = ""
            reverse_complement = ""
            complement_list =[]
            for line in filehandler:
                if not line.startswith('>'):
                    line = line.strip()
                    complement += line
        for i in range(len(complement)-101):
            seq_nucleotide = []
            sequence = complement[i:i+101]
            for nucleotide in sequence:
                seq_nucleotide.append(nucleotide)
            complement_list.append(seq_nucleotide)
        return complement_list, name

    def ANN_model(self, X, y, seq_length, pos_tensor, neg_tensor, Test_pos_tensor, Test_neg_tensor, complement_tensor, name):
        #complement_tensor = sequence_extraction_real(complement_list)
        model = Sequential() #Sequential model (linear stack of layers)
        model.add(Convolution1D(1, 19, border_mode='same', input_shape=(seq_length, 4), activation='relu'))

        #sanity check for dimensions
        #print('Shape of the output of first layer: {}'.format(model.predict_on_batch(pos_tensor[0:32,:,:]).shape))

        #model.add(MaxPooling1D(pool_length=4))
        model.add(Dropout(0.7))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid')) #output layer
        #model.add(LSTM(100))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                      loss='binary_crossentropy', #classfy yes or no
                      metrics=['accuracy']) #output is accuracy

        hist = model.fit(X, y, validation_split=0.3, nb_epoch=10)  # starts training

        
        predictions_pos = model.predict(Test_pos_tensor)
        predictions_neg = model.predict(Test_neg_tensor)
        prediction_test = model.predict(complement_tensor)
        print('pos is ')
        print(predictions_pos)
        print('neg is ')
        print(predictions_neg)
        print('test is ')
        print(prediction_test)
        #print type(prediction_neg)
        positive_prediction = predictions_pos.tolist()
        negative_prediction = predictions_neg.tolist()
        test_prediction = prediction_test.tolist()
        
        #print positive_prediction
        positive_total = 0
        negative_total = 0
        #print len(positive_prediction)
        for i in positive_prediction:
            positive_total += i[0]
        for i in negative_prediction:
            negative_total += i[0]
        count = 1
        output_file = name+'_output.txt'
        with open(output_file, 'w') as output:
            header = 'Position' + '\t' + 'Percent Accuracy' + '\n'
            output.write(header)
            for i in test_prediction:
                line = str(count) + '\t' + str(i[0]) + '\n' 
                output.write(line)
                count +=1
        print(len(positive_prediction), len(negative_prediction))
        print('positive average is: ' + str(positive_total/len(positive_prediction)))
        print('negative average is: ' + str(negative_total/len(negative_prediction)))

        #prediction_comp = model.predict(complement_tensor)


        #have a look at the filter
        convlayer = model.layers[0]
        weights = convlayer.get_weights()[0].transpose().squeeze()
        #print('Convolution parameter shape: {}'.format(weights.shape))

def main():
    instance = Promoter_Prediction()
    X, y, seq_length, pos_tensor, neg_tensor, Test_pos_tensor, Test_neg_tensor = instance.Sequence_extraction()
    fasta = '../res/fabg1G609A.fasta'
    complement_list, name = instance.fasta_sequence(fasta)
    complement_tensor= instance.sequence_extraction_real(complement_list)
    instance.ANN_model(X, y, seq_length, pos_tensor, neg_tensor, Test_pos_tensor, Test_neg_tensor, complement_tensor, name)



if __name__ == "__main__":
    main()

        
