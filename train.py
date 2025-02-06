import torch
from tqdm import tqdm
import wandb
import torchmetrics

from utils import EarlyStopping, get_model, ConfigureResnet

def train(train_dataloader, valid_dataloader, config, sweep_id, sweep_run_name, fold, val_ratio):
    
    #Fold name dict converter
    abc_dict = {'1':"a", '2':'b', '3':'c', '4':'d', '5':'e'}
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f'{sweep_run_name}-{abc_dict[str(fold)]}'
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )


    # Set hyperparameters
    num_epochs = config['epochs']
    batch_size = config['batch_size']
    weight_decay = config['weight_decay']
    learning_rate = config['learning_rate']
    stop_patience = config['stop_patience']
    stop_delta = config['stop_delta']

    #get configured model
    # = get_model(config, device)
    model = ConfigureResnet(architecture = config['architecture'], dropout = config['dropout'],  dropout_rate=0.5)
    model.to(device)

    # Parallelize training across multiple GPUs
    #model = torch.nn.DataParallel(model)

    # Define the loss function, optimizer, lr scheduler, and early stopper
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5])).to(device)
    criterion_eval = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([val_ratio*0.5])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=stop_patience, delta=stop_delta)



# Train the model...
    for epoch in range(num_epochs):
        # initialize metrics
        train_acc_m = torchmetrics.classification.BinaryAccuracy().to(device)
        val_acc_m = torchmetrics.classification.BinaryAccuracy().to(device)
        train_rec_m = torchmetrics.classification.BinaryRecall().to(device)
        val_rec_m = torchmetrics.classification.BinaryRecall().to(device)
        train_prc_m = torchmetrics.classification.BinaryPrecision().to(device)
        val_prc_m = torchmetrics.classification.BinaryPrecision().to(device)
        val_spc_m =  torchmetrics.classification.BinarySpecificity().to(device)

        running_vloss = 0.0
        running_tloss = 0.0

    #  training
    # ----------------------------------------------------------------
        print("Training...")
        for i, (inputs, labels) in tqdm(enumerate(train_dataloader)):
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            train_loss = criterion(outputs, labels.float())
            running_tloss += train_loss.item() * inputs.size(0)

            # Backward pass
            train_loss.backward()
            optimizer.step()

            # metric on current batch
            train_acc = train_acc_m(outputs, labels)
            train_rec = train_rec_m(outputs, labels)
            train_prc = train_prc_m(outputs, labels)
            
        # Calculate and print metrics for every epoch
        avg_train_loss = running_tloss / (i + 1)
        train_acc = train_acc_m.compute()
        train_rec = train_rec_m.compute()
        train_prc = train_prc_m.compute()
        print(f'''Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}''')
        print(f'''Train Accuracy {train_acc.item():.4f}, Train Precision: {train_prc.item():.4f}, Train Recall: {train_rec.item():.4f} \n ''')


    #  evaluation
    # ----------------------------------------------------------------
        model.eval()
        print("Validating...")
        with torch.no_grad():
            #  iterating through batches
            for i, (inputs, labels) in tqdm(enumerate(valid_dataloader)):
                #--------------------------------------
                #  sending images and labels to device
                #--------------------------------------
                inputs = inputs.to(device)
                labels = labels.to(device)

                #--------------------------
                #  making classsifications
                #--------------------------
                output = model(inputs)
                val_loss = criterion_eval(output, labels.float())
                running_vloss += val_loss.item() * inputs.size(0)

                # metric on current batch
                val_acc = val_acc_m(output, labels)
                val_rec = val_rec_m(output, labels)
                val_prc = val_prc_m(output, labels)
                val_spc = val_spc_m(output, labels)
                
            # Calculate and print metrics for every epoch
            avg_val_loss = running_vloss / (i + 1)
            val_acc = val_acc_m.compute()
            val_rec = val_rec_m.compute()
            val_prc = val_prc_m.compute()
            val_spc = val_spc_m.compute()
            val_mac_acc = (val_rec.item()+val_spc.item())/2            
            print(f'''Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}''')
            print(f'''Val Macro Accuracy {val_mac_acc:.4f}, Val Micro Accuracy: {val_acc.item():.4f}''')
            print(f'''Val Recall: {val_rec.item():.4f}, Val Specificity {val_spc.item():.4f}, Val Precision: {val_prc.item():.4f} \n ''')
        

        #Log epoch metrics to wandb
        run.log(dict(epoch = epoch+1, train_accuracy = train_acc, train_precision = train_prc, train_recall = train_rec, train_loss = avg_train_loss,
        val_accuracy=val_acc, val_precision = val_prc, val_recall = val_rec, val_specificty = val_spc, val_macro_accuracy = val_mac_acc, val_loss = avg_val_loss))
        
        #Stop model if val_loss hasn't decrease by more than X within the last X epochs.
        early_stopping(avg_val_loss, val_acc, val_prc, val_rec, val_mac_acc, val_spc)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    #log BEST val metrics from each run to be used in cross val average
    val_dict = {'val_acc': early_stopping.best_acc, 'val_prc':early_stopping.best_prc, 'val_rec': early_stopping.best_rec, 'val_loss': early_stopping.best_score,
                'val_spc': early_stopping.best_spc, 'val_macro_acc': early_stopping.best_macro_acc}
    train_dict = {'train_acc':train_acc, 'train_prc': train_prc, 'train_rec': train_rec, 'train_loss': avg_train_loss}

    import shutil

    # uncomment to remove the training files
    # shutil.rmtree('./annotated_data')
    try:
        shutil.rmtree('./wandb')
        shutil.rmtree('./model_training_checkpoints')
    except:
        pass

    run.finish()

    print(f'Finished Training fold {fold}')
    return(val_dict, train_dict)



