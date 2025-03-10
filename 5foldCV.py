
exec(open("dependencies.py").read())
exec(open("transformerBlocks_Test27.py").read())
np.random.seed(126)  

def train_fold(fold, train_index, test_index, X, y, unique):

    #train test split
    xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
    yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

    #compute feature weights using training data
    feature_weights,pooling_weights= getWeights(xTrain,yTrain,xTest,yTest)


    # Prepare tensors and dataloaders

    xTrain = pd.DataFrame(xTrain)
    yTrain = pd.DataFrame(yTrain)
    xTest = pd.DataFrame(xTest)
    yTest = pd.DataFrame(yTest)
    X_train_tensor = torch.tensor(xTrain.values, dtype=torch.long)
    y_train_tensor = torch.tensor(yTrain.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(xTest.values, dtype=torch.long)
    y_test_tensor = torch.tensor(yTest.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Transformer model definition
    src_vocab_size = int(unique)
    tgt_vocab_size = 1
    d_model = 200
    d_ff = 100
    num_heads = 5
    num_layers = 5
    max_seq_length = X.shape[1]
    dropout = 0.05
    criterion = lambda estimations, batch_y: focalLoss(beta=0.5, gamma=1, batch_y=batch_y, estimations=estimations)
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, feature_weights,pooling_weights)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    transformer.apply(initialize_attention_weights)

    # train loop
    loss_values = []
    for epoch in range(50):
        transformer.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_y = batch_y.squeeze(-1)
            optimizer.zero_grad()
            estimations = transformer(batch_x)
            loss = criterion(estimations, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        loss_values.append(avg_train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Fold {fold}, Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")

    #save training behavior 
    loss_csv_file = f"SY_optim_loss_fold{fold}_CDBN_Test3.csv"
    with open(loss_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss"])
        for i in range(len(loss_values)):
            writer.writerow([i + 1, loss_values[i]])
            
    # Test after training 
    accuracy, values = evaluateModel()
    
    values.to_csv(f"pred_true_SY_fold{fold}_CDBN_Test3.csv")

    print(f"Finished fold {fold}")

    return accuracy


# Load and preprocess the data
data = pd.read_csv("fullDatasetSY.csv")

X,y,unique = preprocess(data)

# Set up cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=100)
cv_results = []

# Use ProcessPoolExecutor to run folds in parallel
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = []
    fold = 1
    for train_index, test_index in kf.split(X):
        futures.append(executor.submit(train_fold, fold, train_index, test_index, X, y, unique))
        fold += 1
    # Collect the results
    for future in concurrent.futures.as_completed(futures):
        cv_results.append(future.result())

# Save accuracies for each fold
mean_value = np.mean(cv_results)  # Calculate the mean
mean_df = pd.DataFrame(mean_value)  # Convert to a DataFrame
mean_df.to_csv("SY_CV_Accuracy_CDBN.csv", index=False)  # Save to CSV           
