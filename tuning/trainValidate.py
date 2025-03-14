''' Load dependencies '''
exec(open("dependencies.py").read())
exec(open("transformerBlocks.py").read())
exec(open("transformerBuild.py").read())

import json
from concurrent.futures import ProcessPoolExecutor

#Load parameters from the file
with open('chosenParams.txt', 'r') as file:
    params = json.load(file)
    trait = params['trait']
    parameter = params['parameter']
    param_values = params['param_values']

print(f"Reloaded Parameters - Trait: {trait}, Parameter: {parameter}, Param Values: {param_values}")

'''Load data'''
if trait == 'Yield':
    data = pd.read_csv("SY_Data.csv")
if trait == '100 seed Weight':
    data = pd.read_csv("SW_Data.csv")
if trait == 'Days to Flowering':
    data = pd.read_csv("DF_Data.csv")
if trait == 'Moisture':
    data = pd.read_csv("SM_Data.csv")
print(f"loaded {trait} data")

X = data.drop(['Unnamed: 0', 'IDS', f'{trait}'], axis=1)
y = data[f'{trait}']

'''find the vocab size '''
stacked = X.stack().unique()
unique = stacked.shape[0]
src_vocab_size = int(unique)
tgt_vocab_size = 1
max_seq_length = X.shape[1]

xTrain, xTest,yTrain, yTest = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)
xTrain, xValid,yTrain, yValid = train_test_split(X, y, 
                                                    test_size=0.5, 
                                                    random_state=0)


xTrain.to_csv(f"{trait}_X.csv")
yTrain.to_csv(f"{trait}_Y.csv")
xTest.to_csv(f"{trait}_X.csv")
yTest.to_csv(f"{trait}_Y.csv")

def train_and_evaluate(value, xTrain, yTrain, xValid, yValid):
    loss_values = []  # Store loss values across all epochs
    
    # Test on Valid Set

    xTrain = torch.tensor(xTrain.values, dtype=torch.int64)
    yTrain = torch.tensor(yTrain.values, dtype=torch.float32)
    xValid = torch.tensor(xValid.values, dtype=torch.int64)
    yValid = torch.tensor(yValid.values, dtype=torch.float32)    

    train_dataset = TensorDataset(xTrain, yTrain)

    if parameter == "Learning Rate":
        lr = value
    else:
        lr = 0.05
    if parameter == "Batch Size":
        batch = value
    else:
        batch =30
    if parameter == "Model Dimension":
        d_model = value
    else:
        d_model=100
    if parameter =="Number of Heads":
        num_heads = value
    else:
        num_heads = 2
    if parameter =="Number of Layers":
        num_layers = value
    else:
        num_layers = 2
    if parameter =="MLP Dimension":
        d_ff = value
    else:
        d_ff = 100
    if parameter == "Dropout Rate":
        dropout = value
    else:
        dropout = 0.05

    batch_size = batch 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    transformer.apply(initialize_attention_weights)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

    for epoch in range(20):
        transformer.train()
        epoch_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            estimations = transformer(batch_x)
            loss = criterion(estimations, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_values.append(avg_epoch_loss)
        torch.save(transformer,f"transformerSY_{parameter}_{value}.pth")
        print(f"finished epoch {epoch} for {parameter} = {value}. Loss: {avg_epoch_loss}")
            
    
    torch.save(transformer, f"transformerSY_{parameter}_{value}_FINAL.pth")
    print("saved model to file")
    
    transformer.eval()
    with torch.no_grad():
        prediction = transformer(xValid)

    prediction_np = prediction.numpy()
    yValid_np = yValid.numpy()
    result, _ = pearsonr(prediction_np, yValid_np)
    
    loss_values = pd.DataFrame({'Epoch': range(1, len(loss_values) + 1), f"Loss_{parameter}_{value}": loss_values})
    predictions_df = pd.DataFrame({'Prediction': prediction_np.tolist(), 'True Value': yValid_np.tolist()})
    
    print(f"finished_{parameter} = {value}")

    return value,result,loss_values, predictions_df

print("defined transformer")
print("ready to begin parallel execution")

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(train_and_evaluate, param_values, [xTrain] * len(param_values),
                                    [yTrain] * len(param_values), [xValid] * len(param_values),
                                    [yValid] * len(param_values)))

    # Save results (parameter vs accuracy)
    results_df = pd.DataFrame([(r[0], r[1]) for r in results], columns=[parameter, 'Accuracy'])
    results_df.to_csv(f"{trait}_{parameter}_Results.csv", index=False)
    print("Results saved")

    # Save loss values for each parameter
    loss_dfs = []
    for value, _, loss_df in results:
        loss_df['Parameter'] = value
        loss_dfs.append(loss_df)
    full_loss_df = pd.concat(loss_dfs, ignore_index=True)
    full_loss_df.to_csv(f"{trait}_{parameter}_{value}_LossValues.csv", index=False)
    print("Loss values saved")

    # Save predictions and true values
    pred_true_dfs = []
    for value, _, _, (predictions, truths) in results:
        df = pd.DataFrame({'Parameter': value, 'Prediction': predictions, 'True Value': truths})
        pred_true_dfs.append(df)
    full_pred_true_df = pd.concat(pred_true_dfs, ignore_index=True)
    full_pred_true_df.to_csv(f"{trait}_{parameter}_{value}_Prediction.csv", index=False)
    print("Predictions and true values saved")



