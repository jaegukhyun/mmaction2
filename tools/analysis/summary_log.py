import sys

def summary_2d_log(log_file):
    best_mAP = [0, 0]
    best_mAP50 = [0, 0]
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if 'bbox_mAP' in line:
                epoch = line.split('[')[1].split(']')[0]
                mAP = float(line.split('bbox_mAP: ')[1].split(',')[0])
                mAP50 = float(line.split('bbox_mAP_50: ')[1].split(',')[0])
                print(epoch, mAP, mAP50)
                if mAP > best_mAP[1]:
                    best_mAP = [epoch, mAP]
                if mAP50 > best_mAP50[1]:
                    best_mAP50 = [epoch, mAP50]
        print(best_mAP)
        print(best_mAP50)

def summary_3d_log(log_file):
    best_mAP = [0, 0]
    catch = False
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if 'mAP@0.5IOU:' in line:
                mAP = line[:-1].split('mAP@0.5IOU:')[1]
                epoch = line.split('[')[1].split(']')[0]
                epoch = int(epoch)
                mAP = float(mAP)
                if mAP > best_mAP[1]:
                    best_mAP = [epoch, mAP]
                    mAP_per_cat = []
                    catch = True
            if catch:
                if 'PerformanceByCategory' in line:
                    mAP_per_cat.append(line[:-1])
                else:
                    if len(mAP_per_cat) > 0:
                        catch = False

    print(best_mAP)
    for line in mAP_per_cat:
        print(line)

if __name__ == '__main__':
    args = sys.argv[1:]
    summary_3d_log(args[0])
