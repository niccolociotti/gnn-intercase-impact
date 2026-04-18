import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from os.path import join, exists
from os import makedirs, listdir
from shutil import rmtree
from itertools import product
from torch.utils.data import Dataset
#from data_encoding import main as data_encoding
from data_encoding_modified import main as data_encoding
from results_evaluation import eval_results, compute_metrics
from config import K, PATIENCE, EPOCHS, BATCH_SIZE, LOG_NAME, VARIANT_TO_TEST, get_grid_combinations, resume_grid_combination


class PrefixDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list.copy()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data_list[idx]


def set_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


if __name__ == '__main__':
    #set_seed()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for variant in VARIANT_TO_TEST:
        for log_name, k in sorted(list(product(LOG_NAME, K))):
            combinations = get_grid_combinations(log_name, variant, k)
            print(f"\n** Device: {device}\n** Variant: {variant}\n** Dataset: {log_name}_{k}_k")
            ts_path = join('dataset', f'{variant}_{log_name}_{k}_k_tensors')
            if len(combinations) == 0:
                eval_results(log_name, variant, k)
                continue
            if not exists(join(ts_path, 'done')):
                data_encoding(log_name, variant, k)
            prefixes = [torch.load(join(ts_path, part), weights_only=False) for part in listdir(ts_path)
                       if part.startswith(f'{log_name}_') and part.endswith('.pt')]
            prefixes = PrefixDataset(prefixes)

            train_dataset = [data for data in prefixes if data.set == 'train']
            test_dataset = [data for data in prefixes if data.set == 'test']
            random.shuffle(train_dataset), random.shuffle(test_dataset)

            train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

            #train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, follow_batch=['x_ctx'],shuffle=True)
            #test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, follow_batch=['x_ctx'])

            for comb_nr, combination in enumerate(combinations):
                comb_path, comb_string, parameters = combination
                makedirs(comb_path, exist_ok=True)

                if variant == 'var_fict_200K_2':
                    from gnn import GNN
                    num_features = prefixes[0].x.shape[-1]

                num_classes = prefixes[0].y.shape[-1]
                model = GNN(num_features=num_features, num_classes=num_classes, parameters=parameters)
                model = model.to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])
                epochs_done, no_improvements, best_loss = 0, 0, np.inf
                df_results = pd.DataFrame()

                if resume_grid_combination(comb_path):
                    # resume interrupted combination
                    print(f"\nResuming combination: {comb_string} ({comb_nr+1}/{len(combinations)})")

                    # loading checkpoint
                    try:
                        checkpoint = torch.load(join(comb_path, 'checkpoint.tar'), weights_only=True)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        epochs_done, no_improvements, best_loss, = checkpoint['epoch'], checkpoint['no_improvements'], checkpoint['best_loss']

                        df_results = pd.read_csv(join(comb_path, f'results_{comb_string}.csv'), header=0, sep=',')
                    except (Exception, ):
                        print(f'Checkpoint not found, skipping combination: {comb_string}')
                        continue

                if epochs_done == 0:
                    print(f"\nStarting combination: {comb_string} ({comb_nr + 1}/{len(combinations)})")

                prefix_pred = []
                criterion = torch.nn.CrossEntropyLoss()
                for epoch in range(epochs_done, EPOCHS):
                    # train model
                    model.train()
                    train_loss = 0.0
                    predictions, labels = torch.tensor([], device=device), torch.tensor([], device=device)
                    active_prefix_size = torch.tensor([], device=device)
                    concurrent_nodes = torch.tensor([], device=device)

                    for batch in train_loader:
                        batch = batch.to(device)
                        active_prefix_size = torch.cat((active_prefix_size, batch.active_prefix_size))
                        concurrent_nodes = torch.cat((concurrent_nodes, batch.concurrent_nodes))
                        out = model(batch)

                        batch_predictions = torch.log_softmax(out, dim=-1).argmax(dim=-1)
                        predictions = torch.cat((predictions, batch_predictions))
                        batch_labels = batch.y.argmax(dim=-1)
                        labels = torch.cat((labels, batch_labels))

                        # train loss
                        loss = criterion(out, batch_labels)

                        # Backpropagation
                        optimizer.zero_grad()  # reset the gradients of all parameters
                        loss.backward()  # compute parameters gradients
                        optimizer.step()  # update parameters

                        train_loss += loss.item()

                    labels, predictions = labels.cpu().tolist(), predictions.cpu().tolist()
                    train_loss /= len(train_loader)
                    train_metrics = compute_metrics(labels, predictions, 'train')

                    df_epoch_train = pd.DataFrame({
                        'set': ['train'] * len(active_prefix_size),
                        'size': active_prefix_size.cpu().tolist(),
                        'prediction': predictions,
                        'label': labels,
                        'concurrent_nodes': concurrent_nodes.cpu().tolist(),
                    })

                    # test model
                    model.eval()
                    test_loss = 0.0
                    predictions, labels = torch.tensor([], device=device), torch.tensor([], device=device)
                    active_prefix_size = torch.tensor([], device=device)
                    concurrent_nodes = torch.tensor([], device=device)

                    with torch.no_grad():
                        for batch in test_loader:
                            batch = batch.to(device)
                            active_prefix_size = torch.cat((active_prefix_size, batch.active_prefix_size))
                            concurrent_nodes = torch.cat((concurrent_nodes, batch.concurrent_nodes))
                            out = model(batch)

                            batch_predictions = torch.log_softmax(out, dim=-1).argmax(dim=-1).int()
                            predictions = torch.cat((predictions, batch_predictions))
                            batch_labels = batch.y.argmax(dim=1)
                            labels = torch.cat((labels, batch_labels.int()))

                            # test loss
                            loss = criterion(out, batch_labels)
                            test_loss += loss.item()

                    # test epoch metrics
                    labels, predictions = labels.cpu().tolist(), predictions.cpu().tolist()
                    test_loss /= len(test_loader)
                    test_metrics = compute_metrics(labels, predictions, 'test')
                    df_epoch_test = pd.DataFrame({
                        'set': ['test'] * len(active_prefix_size),
                        'size': active_prefix_size.cpu().tolist(),
                        'prediction': predictions,
                        'label': labels,
                        'concurrent_nodes': concurrent_nodes.cpu().tolist(),

                    })
                    # model summary
                    if epoch % 5 == 0:
                        print(f"Epoch: {epoch}/{EPOCHS} | Train/test loss {train_loss:.8f}/{test_loss:.8f}")

                    # early stopping
                    if test_loss <= best_loss:
                        no_improvements = 0
                        best_loss = test_loss
                        print(f"** Best test loss: {best_loss:.8f} at epoch: {epoch}")
                        df_epoch_results = pd.concat([df_epoch_train, df_epoch_test])
                        df_epoch_results.to_csv(join(comb_path, 'best_loss_prefix_results.csv'), index=False, header=True, sep=',')
                        torch.save(model.state_dict(), join(comb_path, 'best_loss_prefix_model.pt'))
                    else:
                        no_improvements += 1

                    metrics = {
                        'epoch': epoch,
                        'train_loss': round(train_loss, 8),
                        'test_loss': round(test_loss, 8),
                        'best_test_loss': str(test_loss == best_loss)
                    }
                    metrics.update(train_metrics)
                    metrics.update(test_metrics)
                    df_results = pd.concat([df_results, pd.DataFrame([metrics])], axis="rows")
                    df_results.to_csv(join(comb_path, f'results_{comb_string}.csv'), header=True, index=False, sep=',')

                    # saving checkpoint
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'no_improvements': no_improvements,
                    }, join(comb_path, 'checkpoint.tar'))

                    if no_improvements > PATIENCE:
                        print(f"** Early stopping at epoch: {epoch}")

                        # combination completed
                        with open(join(comb_path, 'done'), 'w') as f:
                            pass
                        break

                # combination completed
                with open(join(comb_path, 'done'), 'w') as f:
                    pass

            rmtree(ts_path)
            eval_results(log_name, variant, k)

