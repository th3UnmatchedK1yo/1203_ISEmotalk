import extract_feature as ef
from tensorflow.keras.utils import to_categorical
import model
from utils import parse_opt
from sklearn.model_selection import KFold
import numpy as np

def train(config,k_folds:int) -> None:
    # Lấy đường dẫn dữ liệu
    data_path = ef.get_path(config.dataset.data_path, train=True)
    x_train, x_test, y_train, y_test = ef.get_features(data_path, config, train=True)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold = 1
    fold_accuracies = []

    for train_index, val_index in kf.split(x_train):

        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        print(f"----- Training fold {fold} -----")


        train_model = model.make_model(config=config, n_feats=x_train_fold.shape[1], num_classes=y_train_fold.shape[1])


        y_train_fold_one_hot = to_categorical(y_train_fold)
        y_val_fold_one_hot = to_categorical(y_val_fold)

        train_model.train(
            x_train_fold, y_train_fold_one_hot,
            x_val_fold, y_val_fold_one_hot,
            config
        )

        fold_acc = train_model.evaluate(x_val_fold, y_val_fold)
        fold_accuracies.append(fold_acc)

        print(f"----- End of fold {fold} -----")
        fold += 1


    mean_acc = np.mean(fold_accuracies)
    print(f"Mean accuracy over {k_folds} folds: {mean_acc:.3f}")

    print("----- Final evaluation on test set -----")
    train_model.evaluate(x_test, y_test)

    # Lưu mô hình
    train_model.save(config.checkpoint.checkpoint_path, config.checkpoint.checkpoint_name)

if __name__ == '__main__':
    config = parse_opt()
    train(config, k_folds=5)