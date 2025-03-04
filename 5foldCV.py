exec(open("dependencies.py").read())
exec(open("transformerBlocks_Test27.py").read())

'''
load data
'''

data = pd.read_csv("fullDatasetSY.csv")
data = data.sample(frac =1)

'''
isolate X and Y
remove any low variance features
they will not contribute to the model
'''

X = data.drop(["Unnamed: 0","0","1","2","3"],axis=1)
threshold = 0.01
X = X.drop(X.std()[X.std() < threshold].index.values, axis=1)

y = data["3"]
IDS = data["2"]


'''
identify number of unique tokens
'''
unique = X.stack().nunique()

'''
fold cross validation
'''
kf = KFold(n_splits=5, shuffle=True, random_state=100)
fold = 1
cv_results = []
true_pred = []

for train_index, test_index in kf.split(X):
    
    
    '''
    train test split
    '''

    xTrain, xTest = X.iloc[train_index], X.iloc[test_index]
    yTrain, yTest = y.iloc[train_index], y.iloc[test_index]

    yValid = yTrain.sample(frac = 0.8)
    xValid = xTrain.sample(frac = 0.8)

    '''determine best alpha value for RR'''

    alphaValues = {}
    params =[0.01,0.1,1,10,100,1000]
    for values in params:
        LR = linear_model.Ridge(values)
        LR.fit(xTrain, yTrain)
        yPred = LR.predict(xValid)
        yValid = np.array(yValid)
        alphaValues[values] = np.corrcoef(yPred, yValid)[0,1]
        alphaVal = max(alphaValues, key=alphaValues.get)

    ''' do RR to get weights'''
    
    RR = linear_model.Ridge(alphaVal)
    RR.fit(xTrain, yTrain)
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

    '''make tensors and create data loader'''
    xTrain = torch.tensor(xTrain.values, dtype=torch.long)
    yTrain = torch.tensor(yTrain.values, dtype=torch.float32)
    xTest = torch.tensor(xTest.values, dtype=torch.long)
    yTest = torch.tensor(yTest.values, dtype=torch.float32)

    train_dataset = TensorDataset(xTrain, yTrain)
    test_dataset = TensorDataset(xTest, yTest)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


    '''
    define transformer parameters
    '''
    src_vocab_size = int(unique)
    tgt_vocab_size = 1
    d_model = 200
    num_heads = 5
    num_layers = 5
    d_ff = 100
    max_seq_length = X.shape[1]
    dropout = 0.05
    criterion = lambda estimations, batch_y: focalLoss(beta=0.5, gamma=1, batch_y=batch_y, estimations=estimations)

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,feature_weights,pooling_weights)


    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    transformer.apply(initialize_attention_weights)
    
    loss_values = []
    val_losses = []
    val_accuracies = []
    
    
    '''
    training loop
    '''
    val_accuracy_prev=0
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

        '''
        test at each epoch
        '''
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
            values.colnames = ["pred","true"]
            val_accuracy_prev = val_accuracy
        
        

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | LR: {current_lr:.6f}")

    '''
    save results for best epoch
    '''
    cv_results.append(max(val_accuracies))


'''
save accuracies at each fold
'''
pd.DataFrame(cv_results).to_csv("SY_CV_Accuracies_Test45.csv")


    
                     
