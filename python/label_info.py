import operator
import os

# path to the label folder
LABEL_ROOT = '/home/dtc/Data/KITTI/data_object_label_2/training/label_2'


def type_of_objects():

    label_stat = {}

    for frame in range(0, 7481):

        label_filename = os.path.join(LABEL_ROOT, '{0:06d}.txt'.format(frame))
        with open(label_filename) as f_label:
            lines = f_label.readlines()
            for line in lines:
                line = line.strip('\n').split()
                label_stat[line[0]] = label_stat.setdefault(line[0], 0) + 1

    return sorted(label_stat.items(), key=operator.itemgetter(1), reverse=True)


if __name__ == '__main__':
    label_stat = type_of_objects()
    print('{0:d} types of objects'.format(len(label_stat)))
    print(label_stat)
