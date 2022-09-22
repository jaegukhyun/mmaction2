import pickle
import numpy as np


def main():
    annot_pkl = ('/home/jaeguk/workspace/data/jester/jester_SC/SC_jester_3cls_12_samples_seed_3/'
                 'annotations/SC_jester_3cls_12_samples_seed_3_dense_proposals_instances_valid.pkl')
    with open(annot_pkl, 'rb') as f:
        info = pickle.load(f, encoding='latin1')
        info['65518,00018'][0][2] = 0.7
        info['77161,00018'][0][0] = 0.0
        info['77161,00018'][0][1] = 0.2
        info['77161,00018'][0][2] = 0.75
        info['77161,00018'][0][3] = 1.0
        info['81063,00019'][0][0] = 0.0
        info['81063,00019'][0][2] = 0.8
        info['94172,00018'][0][0] = 0.0
        _annot_pkl = ('/home/jaeguk/workspace/data/jester/jester_SC/SC_jester_3cls_12_samples_seed_3/'
                 'annotations/SC_jester_3cls_12_samples_seed_3_dense_proposals_instances_valid_mod.pkl')
    with open(_annot_pkl, 'wb') as f:
        pickle.dump(info, f)


if __name__ == '__main__':
    main()
