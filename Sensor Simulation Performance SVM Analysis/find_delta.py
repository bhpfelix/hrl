import util
import matplotlib.pyplot as plt
import numpy as np
import os

matpath = "../simulation_confusion_mats"
pltpath = "../delta_e_plots"
c_dic = {'1.00':'b', '2.40':'g', '3.80':'r', '5.20':'c', '6.60':'m', '8.00':'k'}
markers = {'1.00':'.', '2.40':'1', '3.80':'x', '5.20':'o', '6.60':'*', '8.00':'d'}
matsize = 420
resize_factors = [i for i in range(1,matsize+1) if matsize/i == matsize/(i-0.0)]
MIN_E = 0.089553894
MAX_E = 107.8818103
e_factor = (MAX_E - MIN_E)/float(matsize)
e_range = np.linspace(MIN_E, MAX_E, matsize)


def resize_mat(mat, n):
    m = np.array(mat)
    rows, cols = m.shape

    if (rows/float(n))%1 != 0 or (cols/float(n))%1 != 0:
        raise ValueError("Mat Dimension should be devisible by n.")

    newMatrix = np.zeros([rows/n, cols/n])

    for i in range(rows):
        for j in range(cols):
            newMatrix[i/n][j/n] += m[i][j]

    return newMatrix

# Not that good because when interval is small,
# the score will be inflated by the large proportion
# of true negatives
def accuracy_score(m):
    rows, cols = m.shape
    scores = np.zeros(rows)

    incidents = np.sum(m)
    for i in range(rows):
        tp = m[i][i]
        fp = np.sum(m, axis=0)[i] - m[i][i]  #The corresponding column for class_i - TP
        fn = np.sum(m, axis=1)[i] - m[i][i] # The corresponding row for class_i - TP
        tn = incidents - tp - fp - fn

        accuracy = (tp+tn)/float(incidents)
        scores[i] = accuracy

    return scores

def f1_score(m):
    rows, cols = m.shape
    scores = np.zeros(rows)

    incidents = np.sum(m)
    for i in range(rows):
        tp = m[i][i]
        fp = np.sum(m, axis=0)[i] - m[i][i]  #The corresponding column for class_i - TP
        fn = np.sum(m, axis=1)[i] - m[i][i] # The corresponding row for class_i - TP
        tn = incidents - tp - fp - fn

        # How likely is an obj to be correctly classified apart from other
        precision = (float(tp) / (tp + fp)) if (tp + fp) != 0 else 0
        # How likely is an obj to be correctly classified as itself
        recall = (float(tp) / (tp + fn)) if (tp + fn) != 0 else 0
        scores[i] = (2*precision*recall / (precision + recall)) if (precision + recall) != 0 else 0

    return scores

if __name__ == '__main__':
    mat_dics = {}
    for f in os.listdir(matpath):
        if f.startswith("confusion_matrices_"):
            time = f[:-4].split('_')[-1]
            mat_dics[time] = util.load_pickle(os.path.join(matpath, f))

    times = mat_dics.keys()
    models = mat_dics[times[0]].keys()

    for model in models:
        fig, ax = plt.subplots()
        plt.title("t_amb = %s, t_sens_0 = %s, noise = %s" % (model[0], model[1], model[2]))
        plt.xlabel('e')
        plt.ylabel('delta_e')
        ymin = 0
        ymax = 0

        for time in times:
            delta_e = [[430] for _ in range(matsize)]
            for fac in resize_factors:
                m = resize_mat(mat_dics[time][model], fac)
                score = f1_score(m)

                for i in range(matsize):
                    s = score[i/fac]
                    if s >= 0.9:
                        delta_e[i].append(fac)

            X = []
            Y = []

            for ind, l in enumerate(delta_e):
                X.append(ind*e_factor)
                Y.append(min(l))

            ax.plot(X, Y, color=c_dic[time], label = str(time) + 's')

            ymax = max(ymax, max(Y))

        ymax = ymax + (ymax-ymin)*0.1
        plt.gca().set_ylim([ymin, ymax])

        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, loc='upper right', shadow=True)

        plt.savefig("%s/%s_%s_%s.png" % (pltpath, model[0], model[1], model[2]))

    plt.show()