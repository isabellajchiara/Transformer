
exec(open("dependencies.py").read())
exec(open("transformerBlocks_Test27.py").read())
import concurrent.futures

data = pd.read_csv("fullDatasetSY.csv")

# Define the function that will process each test proportion

def process_prop(prop, X, y, data, batch_size=30):
    cv_results = []
    true_pred = []

    x = slice(0, int(round(data.shape[0] * prop, 0)))

    trainingSet = data.iloc[0:int(round(data.shape[0] * 0.6, 0)), :]

    testSet = data.iloc[x, :]
    
    threshold = 0.01

    trainingSet = data.iloc[0:int(round(data.shape[0] * 0.6, 0)), :]
    testSet = data.iloc[x, :]
    x = slice(0, int(round(data.shape[0] * prop, 0)))

    xTrain = trainingSet.drop(["Unnamed: 0", "0", "1", "2", "3"], axis=1)
    xTrain = xTrain.drop(xTrain.std()[xTrain.std() < threshold].index.values, axis=1)
    yTrain = trainingSet["3"]

    xTest = testSet.drop(["Unnamed: 0", "0", "1", "2", "3"], axis=1)
    xTest = xTest.drop(xTest.std()[xTest.std() < threshold].index.values, axis=1)
    yTest = testSet["3"]

    unique = xTrain.stack().nunique()

    # Determine best alpha value for Ridge Regression (RR)
    alphaValues = {}
    params = [0.01, 0.1, 1, 10, 100, 1000]
    for values in params:
        LR = linear_model.Ridge(values)
        LR.fit(xTrain, yTrain)
        yPred = LR.predict(xValid)
        yValid = np.array(yValid)
        alphaValues[values] = np.corrcoef(yPred, yValid)[0, 1]
        alphaVal = max(alphaValues, key=alphaValues.get)

    # Perform Ridge Regression to get weights
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
    
    # Convert to tensors
    xTrain = torch.tensor(xTrain.values, dtype=torch.float32)
    yTrain = torch.tensor(yTrain.values, dtype=torch.float32)
    xTest = torch.tensor(xTest.values, dtype=torch.float32)
    yTest = torch.tensor(yTest.values, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(xTrain, yTrain)
    test_dataset = TensorDataset(xTest, yTest)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize the model, optimizer, and scheduler
    src_vocab_size = int(unique)
    tgt_vocab_size = 1
    d_model = 200
    num_heads = 5
    num_layers = 5
    d_ff = 100
    max_seq_length = xTrain.shape[1]
    dropout = 0.05
    criterion = lambda estimations, batch_y: focalLoss(beta=0.5, gamma=1, batch_y=batch_y, estimations=estimations)

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, feature_weights,pooling_weights)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    transformer.apply(initialize_attention_weights)

    # Training loop
    loss_values, val_losses, val_accuracies = [], [], []
    val_accuracy_prev = 0
    for epoch in range(40):
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

        # Validation loop
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
            values = pd.DataFrame(values, columns=["pred", "true"])
            values.to_csv(f"pred_true_SY_CDBN_{prop}.csv")
            val_accuracy_prev = val_accuracy

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | LR: {current_lr:.6f}")

    # Save the results for this prop
    cv_results.append(max(val_accuracies))
    torch.save(transformer, f"transformerSY_CDBN_{prop}.pth")
    loss_csv_file = f"SY_optim_loss_CDBN_{prop}.csv"
    with open(loss_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy"])
        for i in range(len(loss_values)):
            writer.writerow([i + 1, loss_values[i], val_losses[i], val_accuracies[i]])

    return cv_results



# Prepare the list of props
testProportions = [0.6, 0.7, 0.8, 0.9, 0.99]

# Initialize the executor to run processes in parallel
cv_results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = []
    for prop in testProportions:
        # Map the process_prop function to each prop
        futures.append(executor.map(lambda prop: process_prop(prop, X, y, data), testProportions))
    # Collect the results
    for future in concurrent.futures.as_completed(futures):
        cv_results.append(future.result())

# Aggregate or save the results after all tasks are completed
pd.DataFrame(cv_results).to_csv("SY_CV_Accuracies_CDBN_testProps.csv")
