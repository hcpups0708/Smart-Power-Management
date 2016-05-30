import os

base_path = '/home/spark/mhwong/cloud/project/result'

for id_dir in os.listdir(base_path):
    # id directory naming convention: ID.result, eg DG10000421.result
    id = id_dir.split(".")[0]

    # read result file
    result_file = open(os.path.join(base_path, id_dir, "part-00000"))
    num_correct = 0.0
    num_incorrect = 0.0
    for line in result_file:
        # line format (truth,prediction)
        splitted_line = line.split(",")
        truth = float(splitted_line[0].replace('(', ''))
        prediction = float(splitted_line[1].replace(')', ''))

        # count as true if the difference is less than 3
        if abs(truth - prediction) <= 3.0:
            num_correct += 1
        else:
            num_incorrect += 1

    accuracy = (num_correct*1.0) / ((num_correct+num_incorrect)*1.0)
    print id, accuracy
