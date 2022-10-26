from torch.utils.data import Dataset
import os
import numpy as np
import json
from tqdm import tqdm
import pickle
import copy

class mel_dataset(Dataset):

    def __init__(self, data_dir, top_genre):
        
        super(mel_dataset, self).__init__()
        meta_file_path = os.path.join(data_dir, "adjust_song_meta.json")
        if os.path.isfile(meta_file_path):
            with open(meta_file_path) as f:
                song_meta = json.load(f)
        else:
            raise FileNotFoundError(f'No such file or directory: {data_dir}/adjust_song_meta.json')

        song_dict = {}
        genre_dict = {}
        print('Load song_meta.json...')
        for song in tqdm(song_meta):
            song_dict[str(song['id'])] = song['song_gn_gnr_basket']
            for i in song['song_gn_gnr_basket']:
                try:
                    genre_dict[i] += 1
                except:
                    genre_dict[i] = 0
        del genre_dict['GN9000'] # NAN
        self.genre_keys = {k:v for k,v in enumerate(list(genre_dict.keys()))}
        self.genre_index = {k:v for v,k in enumerate(list(genre_dict.keys()))}

        result_dict = {}        
        print('Load complete!')
        print('\nLoad file list...')
        for roots, dirs, files in tqdm(os.walk(data_dir)):
                  
            listdir = [os.path.join(roots, file) for file in files]
            for i in listdir:
                
                if ".npy" in i:
                    b = np.load(i)
                    if b.shape[1] != 1876:
                        pass
                    else: 
                        try:
                            song_id = i.split('/')[-1].replace('.npy','')
                            result_dict[i] = song_dict[song_id]
                        except:
                            print(song_id,'passed.')
                            
        file_list = []
        label = []
        count_dict = {}
        
        for song_id, genres in result_dict.items():
            if len(genres) == 1:
                for value in genres:                    
                    try:
                        count_dict[value] += 1
                    except:
                        count_dict[value] = 0
                        
        count_dict = sorted(count_dict.items(), key = lambda item: item[1], reverse = True)
        self.count_dict = {i[0]:i[1] for i in count_dict}
        self.sort_dict = {}
        
        for n, genre in enumerate(self.count_dict):
            if n == top_genre:
                break
            self.sort_dict[genre] = self.count_dict[genre]            
        self.sort_dict = {k:v for v,k in enumerate(self.sort_dict)}
        for song_id, genres in result_dict.items():
            if len(genres) == 1:
                if top_genre == 'total':
                    one_hot_zero = np.zeros(len(self.genre_index))   
                    for value in genres:     
                        if value != 'GN9000': # pass NAN value
                            one_hot_zero[self.genre_index[value]] = 1
                    file_list.append(song_id)
                    label.append(one_hot_zero)
    
                else: 
                    if top_genre >= len(self.genre_index):
                        raise ValueError(f"There's no {top_genre} index genre. Reduce genre index number.")                                                                                      
                    for value in genres:    
                        # print(value)
                        if value in list(self.sort_dict.keys()):                                                         
                            one_hot_zero = np.zeros(len(self.sort_dict))  
                            one_hot_zero[self.sort_dict[value]] = 1
                            file_list.append(song_id)
                            label.append(one_hot_zero)
            else:
                pass


        self.file_list = file_list
        self.label = label
        
    def __getitem__(self, index):
        self.x = np.load(self.file_list[index])
        return self.x, self.label[index]
    
    def __len__(self):
        return len(self.file_list)
    
    def genre_index(self):
        return self.genre_index
