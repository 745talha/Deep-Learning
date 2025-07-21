import time
import torch
from .helper_evaluation import compute_accuracy
from .helper_evaluation import compute_epoch_loss_autoencoder
from torch.nn import functional as F


# def train_model(model, num_epochs, train_loader,
#                 valid_loader, test_loader, optimizer, device):

#     start_time = time.time()
#     minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
#     for epoch in range(num_epochs):

#         model.train()
#         for batch_idx, (features, targets) in enumerate(train_loader):

#             features = features.to(device)
#             targets = targets.to(device)

#             # ## FORWARD AND BACK PROP
#             logits = model(features)
#             loss = torch.nn.functional.cross_entropy(logits, targets)
#             optimizer.zero_grad()

#             loss.backward()

#             # ## UPDATE MODEL PARAMETERS
#             optimizer.step()

#             # ## LOGGING
#             minibatch_loss_list.append(loss.item())
#             if not batch_idx % 50:
#                 print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
#                       f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
#                       f'| Loss: {loss:.4f}')

#         model.eval()
#         with torch.no_grad():  # save memory during inference
#             train_acc = compute_accuracy(model, train_loader, device=device)
#             valid_acc = compute_accuracy(model, valid_loader, device=device)
#             print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
#                   f'| Train: {train_acc :.2f}% '
#                   f'| Validation: {valid_acc :.2f}%')
#             train_acc_list.append(train_acc.item())
#             valid_acc_list.append(valid_acc.item())

#         elapsed = (time.time() - start_time)/60
#         print(f'Time elapsed: {elapsed:.2f} min')

#     elapsed = (time.time() - start_time)/60
#     print(f'Total Training Time: {elapsed:.2f} min')

#     test_acc = compute_accuracy(model, test_loader, device=device)
#     print(f'Test accuracy {test_acc :.2f}%')

#     return minibatch_loss_list, train_acc_list, valid_acc_list


def train_model(model, num_epochs, train_loader,
                valid_loader, test_loader, optimizer,
                device, logging_interval=50,
                scheduler=None,
                scheduler_on='valid_acc'):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')

        model.eval()
        with torch.no_grad():  # save memory during inference
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc = compute_accuracy(model, valid_loader, device=device)
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')
        
        if scheduler is not None:

            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
                my_last_lr=scheduler.get_last_lr()
                print("Last lr: ",my_last_lr)
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')
        

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list



def train_autoencoder(num_epochs, model, optimizer, device, 
                         train_loader, loss_fn=None,
                         logging_interval=100, 
                         skip_epoch_stats=False,
                         save_model=None):
    
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}
    
    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            features = features.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            loss = loss_fn(logits, features)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict