from simpleRNN import simplernn
from IMDbExample import dataprepare as dp

if __name__ == "__main__":

    #simplernn.simplernn_eval()

    train_data, valid_data, test_data = dp.create_dataset()
    