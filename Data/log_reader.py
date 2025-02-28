import csv

pdoa_data = []
tdoa_data = []
rx_true_r = []

dist_meas = []
twr_count = []
dist_true_r = []

with open("Data/log.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        linetype = int(line[0][0])
        if (linetype == 0):
            # rx data, featuring pdoa, tdoa and distance
            pdoa_data += [float(line[0][1])]
            tdoa_data += [int(line[0][2])]
            rx_true_r = [int(line[0][3])]
        
        if (linetype == 1):
            dist_meas += [int(line[0][4])]
            twr_count += [int(line[0][5])]
            dist_true_r += [int(line[0][6])]

        if (linetype == 2):
            print("CIR data")
            

