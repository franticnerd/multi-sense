import pandas as pd
from torch.utils.data import Dataset, DataLoader

class RelationDataset(Dataset):

    def __init__(self, data_file):
        self.instances = pd.read_table(data_file, header=None)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        label = [self.instances.ix[idx, 0]]
        input = map(int, str(self.instances.ix[idx, 1]).strip().split())
        return input, label



class Vocab():

    def __init__(self, data_file):
        self.id_to_description = {}
        with open(data_file, 'r') as fin:
            for line in fin:
                items = line.strip().split('\t')
                idx = int(items[0])
                description = items[1]
                self.id_to_description[idx] = description

    def size(self):
        return len(self.id_to_description)



def load_data(data_file):
    data = RelationDataset(data_file=data_file)
    return data

# start = time.time()
# dataloader = DataLoader(data, batch_size=4, shuffle=False, num_workers=2)
# for i_batch, sample_batched in enumerate(dataloader):
#     if i_batch == 2:
#         print str(sample_batched)
#         print len(sample_batched)
# end = time.time()
# print end - start


