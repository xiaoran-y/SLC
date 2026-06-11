# Code adapted from https://github.com/pykt-team/pykt-toolkit
import numpy as np
import math



class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen
    # data format
    # id, true_student_id
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        qa_data = []
        idx_data = []
        count_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 0:
                tokens = line.split(self.separate_char)
                try:
                    student_id = int(tokens[0])
                except (ValueError, TypeError, IndexError):
                    student_id = lineID//3
                try:
                    base_offset = int(tokens[2]) if len(tokens) > 2 else 0
                except (ValueError, TypeError):
                    base_offset = 0
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len(Q[len(Q)-1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    count_sequence = []
                    if k == n_split - 1:  # last chunk
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            # int(A[i]) is in {0,1}
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                            count_sequence.append(base_offset + i + 1)
                        else:
                            print(Q[i])
                    #print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    idx_data.append(student_id)
                    count_data.append(count_sequence)
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training

        # Let each row of q_data  = q_dataArray  ,  and remaining = 0
        q_dataArray = np.zeros((len(q_data), self.seqlen), dtype=np.int64)
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen), dtype=np.int64)
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat
        # count is used as a numeric time/index feature; float32 is sufficient and faster to move to GPU.
        count_dataArray = np.zeros((len(count_data), self.seqlen), dtype=np.float32)
        for j in range(len(count_data)):
            dat = count_data[j]
            count_dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_dataArray, qa_dataArray, np.asarray(idx_data, dtype=np.int64), count_dataArray


class PID_DATA(object):
    def __init__(self, n_question,  seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_question = n_question
    # data format
    # id, true_student_id
    # pid1, pid2, ...
    # 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    # 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        qa_data = []
        p_data = []
        idx_data = []
        count_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 4 == 0:
                tokens = line.split(self.separate_char)
                try:
                    student_id = int(tokens[0])
                except (ValueError, TypeError, IndexError):
                    student_id = lineID//4
                try:
                    base_offset = int(tokens[2]) if len(tokens) > 2 else 0
                except (ValueError, TypeError):
                    base_offset = 0
            if lineID % 4 == 2:
                Q = line.split(self.separate_char)
                if len(Q[len(Q)-1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            if lineID % 4 == 1:
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]

            elif lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    problem_sequence = []
                    answer_sequence = []
                    count_sequence = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0:
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            problem_sequence.append(int(P[i]))
                            answer_sequence.append(Xindex)
                            count_sequence.append(base_offset + i + 1)
                        else:
                            print(Q[i])
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    p_data.append(problem_sequence)
                    idx_data.append(student_id)
                    count_data.append(count_sequence)

        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen), dtype=np.int64)
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen), dtype=np.int64)
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen), dtype=np.int64)
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat
        count_dataArray = np.zeros((len(count_data), self.seqlen), dtype=np.float32)
        for j in range(len(count_data)):
            dat = count_data[j]
            count_dataArray[j, :len(dat)] = dat
        return q_dataArray, qa_dataArray, p_dataArray, np.asarray(idx_data, dtype=np.int64), count_dataArray
