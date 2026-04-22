import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset


class GameLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=2):
        super(GameLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 初始化训练引擎... 设备: {device}")

    dataset = TrajectoryDataset('training_data.csv', window_size=20, horizon=5, is_train=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GameLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], MSE Loss: {total_loss / len(dataloader):.6f}')

    torch.save(model.state_dict(), 'lstm_model.pth')
    print("✅ 权重已保存至 lstm_model.pth")


if __name__ == "__main__":
    train_model()
