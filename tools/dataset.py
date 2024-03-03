import sys

import torch
import glob
import numpy as np
import torch.utils.data
from PIL import Image
from torchvision import transforms
from .tool import read_tu_data
from itertools import repeat
from sklearn.model_selection import train_test_split


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, m_dir=None, s_dir=None):
        if m_dir is not None and s_dir is not None:
            self.m_init(m_dir)
            self.s_init(s_dir)

    def m_init(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
               num_models=0, num_views=12, shuffle=False):
        self.classnames = ['00_Gear', '01_Washer', '02_Steel', '03_Nut', '04_Screw', '05_Spring', '06_Bearing',
                           '07_Flange',
                           '08_Ball', '09_Bolt', '10_Elbow', '11_Grooved_Pin', '12_Stud', '13_Round_Nut',
                           '14_Lock_Washer',
                           '15_Bevel_Gear', '16_Helical_Gear', '17_Key', '18_BearingHouse', '19_Distributor',
                           '20_HeaveTightCouplingSleeve',
                           '21_TemplateAndPlate', '22_LiftingHook', '23_WireTensioner', '24_ForgedShackle',
                           '25_SplineEndWrenches',
                           '26_BoringBar', '27_KeylessDrillChuck', '28_HydraulicComponent', '29_BallValve']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views

        # set_ = root_dir.split('/')[-1]
        # parent_dir = root_dir.rsplit('/', 2)[0]
        parent_dir = root_dir
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/*.png'))
            all_files = [s.replace('\\', '/') for s in all_files]
            ## Select subset for different number of views
            stride = int(12 / self.num_views)  # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models, len(all_files))])

        if shuffle == True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths) / num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
            self.filepaths = filepaths_new

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def s_init(self, root_dir):
        self.x, self.edge_index, self.y, self.name, self.name_dict, self.slices, self.num_nodes, self.edge_attr = read_tu_data(root_dir)
    def __len__(self):
        return int(len(self.filepaths) / self.num_views)

    def __getitem__(self, idx):
        # m_dataset
        path = self.filepaths[idx * self.num_views]
        class_name = path.split('/')[-2]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx * self.num_views + i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        # s_dataset
        keys = ['edge_index', 'x', 'y', 'edge_attr', 'name']
        for key in keys:
            slices = self.slices[key]
            # print(slices)
            if key == 'edge_index':
                item = self.edge_index
            elif key == 'x':
                item = self.x
            elif key == 'y':
                item = self.y
            elif key == 'edge_attr':
                item = self.edge_attr
            else:
                item = self.name

            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                # edge_index=-1, y=0, name=0, x=0
                if key == 'edge_index':
                    cat_dim = -1
                else:
                    cat_dim = 0
                s[cat_dim] = slice(start, end)

            if key == 'edge_index':
                edge_index = self.edge_index[s]
            elif key == 'x':
                x = self.x[s]
            elif key == 'y':
                y = self.y[s]
                batch = self.num_nodes.clone().detach()[s].tolist()
            elif key == 'edge_attr':
                edge_attr = self.edge_attr[s]
            else:
                name = self.name[s]

        return (class_id, torch.stack(imgs), self.filepaths[idx * self.num_views:(idx + 1) * self.num_views],
                edge_index, x, y, name, batch, edge_attr)

    def get_train_test_dataset(self):
        total = self.y.shape[0]
        total_index = [index for index in range(total)]
        train_index, test_index = train_test_split(total_index, test_size=0.2)
        train_index.sort()
        test_index.sort()
        train_dataset = self.get_sub_dataset(train_index)
        test_dataset = self.get_sub_dataset(test_index)
        # print(train_dataset)
        # sys.exit(0)
        return train_dataset, test_dataset

    def get_sub_dataset(self, indexes):

        dataset = MultiModalDataset()
        dataset.filepaths, dataset.root_dir, dataset.classnames = [], self.root_dir, self.classnames #处理视图数据
        dataset.num_views, dataset.transform = self.num_views, self.transform
        dataset.rot_aug, dataset.scale_aug, dataset.test_mode = self.rot_aug, self.scale_aug, self.test_mode
        dataset.name_dict = self.name_dict  # 处理step数据
        dataset.y, dataset.x, dataset.name, dataset.edge_index, dataset.edge_attr= [], [], [], [], []
        edge_index, edge_attr, x, y, name, num_nodes = [0], [0], [0], [0], [0], []
        idx = 1
        for index in indexes:
            dataset.y.append(self.y[index])
            dataset.name.append(self.name[index])
            num_nodes.append(self.num_nodes[index])
            for i in range(index * 12, index * 12 + 12):
                dataset.filepaths.append(self.filepaths[i])
            for i in range(self.slices['edge_index'][index], self.slices['edge_index'][index + 1]):
                dataset.edge_index.append(torch.tensor([self.edge_index[0][i], self.edge_index[1][i]]))
                dataset.edge_attr.append(torch.tensor(self.edge_attr[i]))
            edge_index.append(len(dataset.edge_index))
            edge_attr.append(len(dataset.edge_attr))
            for i in range(self.slices['x'][index], self.slices['x'][index + 1]):
                dataset.x.append(self.x[i])
            x.append(len(dataset.x))
            y.append(idx)
            name.append(idx)
            idx += 1


        dataset.num_nodes = torch.tensor(num_nodes)
        dataset.edge_index = torch.stack(dataset.edge_index, -1)
        dataset.edge_attr = torch.stack(dataset.edge_attr, dim=0)
        dataset.x = torch.stack(dataset.x)
        dataset.y = torch.tensor(dataset.y)
        dataset.name = torch.tensor(dataset.name)
        # print(dataset.edge_attr)
        dataset.slices = {'edge_index': torch.tensor(edge_index), 'x': torch.tensor(x), 'y': torch.tensor(y),
                          'name': torch.tensor(name), 'edge_attr': torch.tensor(edge_attr)}

        return dataset
