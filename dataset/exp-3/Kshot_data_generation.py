import pandas as pd


for file in ["train_1187.csv", "valid_1187.csv"]:
    train = pd.read_csv(file)
    type = file.split('_')[0]

    pos_train = train[train['label']==0]
    false_train = train[train['label']==1]
    print(file, 'pos_train',len(pos_train))
    print(file, 'false_train', len(false_train))

    for seed in [4,13,34]:
        for k in [512]:
            print(seed, k)
            pos_samples = pos_train.sample(k, random_state=seed)
            false_samples = false_train.sample(k, random_state=seed)
            kshot_train = pd.concat([pos_samples,false_samples],axis=0)
            kshot_train.to_csv(f'{type}_{k}shots_{seed}seed.csv', index=None)