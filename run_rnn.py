import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CATHDataset
from utils import read_config, MetricTracker, Timer, parse_args


class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, x = self.rnn(x)
        x = self.fc(x.squeeze(0))
        return x

def main():
    args = parse_args()
    config = read_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in range(10):
        train_dataset = CATHDataset("train", fold, config["name"]) 
        test_dataset = CATHDataset("test", fold, config["name"])

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = RNN(input_dim=3, hidden_dim=128, num_classes=config["num_classes"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        tracker = MetricTracker()
        
        for epoch in range(args.num_epochs):
            train_loss = 0.0
            train_acc = 0.0
            test_loss = 0.0
            test_acc = 0.0

            model.train()
            for batch in train_dataloader:
                x, y = batch
                x = x.to(device)
                y = y.to(device).squeeze(0)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += (output.argmax(dim=1) == y).sum().item()

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)

            model.eval()
            with torch.no_grad():
                for batch in test_dataloader:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device).squeeze(0)
                    output = model(x)
                    loss = criterion(output, y)
                    test_loss += loss.item()
                    test_acc += (output.argmax(dim=1) == y).sum().item()    

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

            tracker.update("train_loss", train_loss)
            tracker.update("train_acc", train_acc)
            tracker.update("test_loss", test_loss)
            tracker.update("test_acc", test_acc)

            print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")

        tracker.to_json(f"results/{config['name']}_fold_{fold}.json")
        
if __name__ == "__main__":
    main()