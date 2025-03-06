#batch 5, large LR, reduce for last 5

exec(open("dependencies.py").read())
exec(open("transformerBlocks_Test27.py").read())
import concurrent.futures
np.random.seed(126)  

def train_fold(fold, train_index, test_index, X, y, unique):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]



    # Ridge regression to get the best alpha value
    alpha_values = {}
    params = [0.01, 0.1, 1, 10, 100, 1000]
    for value in params:
        LR = linear_model.Ridge(value)
        LR.fit(X_train, y_train)
        y_pred = LR.predict(X_test)
        y_test = np.array(y_test)
        alpha_values[value] = np.corrcoef(y_pred, y_test)[0, 1]
        alpha_val = max(alpha_values, key=alpha_values.get)

    # Ridge regression to get the coefficients (feature weights)
    RR = linear_model.Ridge(alpha_val)
    RR.fit(X_train, y_train)
    coeffs = RR.coef_
    coeffs = pd.DataFrame(coeffs).squeeze(1)
    min_val = coeffs.min()
    max_val = coeffs.max()
    scaled_coeffs = (coeffs - min_val) / (max_val - min_val)
    

    feature_weights = torch.tensor(scaled_coeffs, dtype=torch.float32)
    pooling_weights = torch.tensor(coeffs,dtype=torch.float32)
    batch_size = 5
    feature_weights = feature_weights.view(1, -1).expand(batch_size, -1)
    pooling_weights = pooling_weights.view(1,-1).expand(batch_size,-1)

    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    
    # Prepare tensors and dataloaders
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

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

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    transformer.apply(initialize_attention_weights)

    loss_values = []
    val_losses = []
    val_accuracies = []

    # Training loop
    val_accuracy_prev = 0
    for epoch in range(30):
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

        # Test after each epoch
        transformer.eval()
        val_loss = 0
        predictions = []
        true_vals = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_y = batch_y.squeeze(-1)
                preds = transformer(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()

                predictions.extend(preds.numpy())
                true_vals.extend(batch_y.numpy())

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        val_accuracy, _ = pearsonr(predictions, true_vals)
        val_accuracies.append(val_accuracy)

        if val_accuracy > val_accuracy_prev:
            values = np.column_stack((predictions, true_vals))
            values = pd.DataFrame(values)
            values.columns = ["pred", "true"]
            values.to_csv(f"pred_true_SY_fold{fold}_CDBN_Test3.csv")
            val_accuracy_prev = val_accuracy

           


        current_lr = optimizer.param_groups[0]['lr']
        print(f"Fold {fold}, Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | LR: {current_lr:.6f}")
        
    # Save results for the best epoch
    fold_result = max(val_accuracies)

    loss_csv_file = f"SY_optim_loss_fold{fold}_CDBN_Test3.csv"
    with open(loss_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy"])
        for i in range(len(loss_values)):
            writer.writerow([i + 1, loss_values[i], val_losses[i], val_accuracies[i]])

    print(f"Finished fold {fold}")

    return fold_result


# Load and preprocess the data
data = pd.read_csv("fullDatasetSY.csv")
data = data.sample(frac=1)


X = data.drop(["Unnamed: 0", "0", "1", "2", "3"], axis=1)
threshold = 0.01
X = X.drop(X.std()[X.std() < threshold].index.values, axis=1)

y = data["3"]
IDS = data["2"]

# Identify the number of unique tokens
unique = X.stack().nunique()

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
mean_df = pd.DataFrame([mean_value], columns=["Mean"])  # Convert to a DataFrame
mean_df.to_csv("SY_CV_Accuracies_CDBN_Test3.csv", index=False)  # Save to CSV


    
                     
