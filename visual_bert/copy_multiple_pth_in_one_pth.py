import torch
import os
output_dir = '../data/VisualBert_blip_large'

def load_and_preprocess_data(split):

    count = 15
    _tmp = []
    if count > 1:
        for i in range(1, count):
            _tmp.append(torch.load(os.path.join(output_dir, str(i) + split + '.pth'), map_location='cpu'))
            print('i',i)
        _tmp.append(torch.load(os.path.join(output_dir, 'final' + split + '.pth'), map_location='cpu'))
        res = torch.cat(_tmp).cpu()
        #print('feature shape:', res.shape, ',save in:', output_dir + '/' + split + '.pth', flush=True)
        torch.save(res, os.path.join(output_dir, split + '.pth'), map_location='cpu')

        # delete
        for i in range(1, count):
            os.remove(os.path.join(output_dir, str(i) + split + '.pth'))
        os.remove(os.path.join(output_dir, 'final' + split + '.pth'))


load_and_preprocess_data('train')
