import math
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard, LambdaCallback


# ================= LR COMPATIBILITY HELPERS =================
def get_lr(optimizer):
    if hasattr(optimizer, "learning_rate"):
        return K.get_value(optimizer.learning_rate)
    else:
        return K.get_value(optimizer.lr)

def set_lr(optimizer, lr):
    if hasattr(optimizer, "learning_rate"):
        K.set_value(optimizer.learning_rate, lr)
    else:
        K.set_value(optimizer.lr, lr)
# ============================================================


# --------------------------------------- DATA PRE-PROCESSING ---------------------------------------
def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'),
                            left_on='unit_nr', right_index=True)
    result_frame["RUL"] = result_frame["max_cycle"] - result_frame["time_cycles"]
    return result_frame.drop("max_cycle", axis=1)

def add_operating_condition(df):
    df_op_cond = df.copy()
    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))
    df_op_cond['op_cond'] = (
        df_op_cond['setting_1'].astype(str) + '_' +
        df_op_cond['setting_2'].astype(str) + '_' +
        df_op_cond['setting_3'].astype(str)
    )
    return df_op_cond

def condition_scaler(df_train, df_test, sensor_names):
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = \
            scaler.transform(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = \
            scaler.transform(df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_train, df_test

def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    df[sensors] = df.groupby('unit_nr')[sensors] \
        .apply(lambda x: x.ewm(alpha=alpha).mean()) \
        .reset_index(level=0, drop=True)

    def create_mask(data, samples):
        mask = np.ones_like(data)
        mask[:samples] = 0
        return mask

    mask = df.groupby('unit_nr')['unit_nr'] \
        .transform(create_mask, samples=n_samples).astype(bool)
    return df[mask]

def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    for start in range(0, data.shape[0] - sequence_length + 1):
        yield data[start:start + sequence_length, :]

def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size == 0:
        unit_nrs = df['unit_nr'].unique()
    data = [
        seq for unit in unit_nrs
        for seq in gen_train_data(df[df['unit_nr'] == unit],
                                  sequence_length, columns)
    ]
    return np.array(data, dtype=np.float32)

def gen_labels(df, sequence_length, label):
    return df[label].values[sequence_length - 1:, :]

def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size == 0:
        unit_nrs = df['unit_nr'].unique()
    labels = [
        gen_labels(df[df['unit_nr'] == unit], sequence_length, label)
        for unit in unit_nrs
    ]
    return np.concatenate(labels).astype(np.float32)

def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        padded = np.full((sequence_length, len(columns)), mask_value)
        padded[-df.shape[0]:] = df[columns].values
        yield padded
    else:
        yield df[columns].values[-sequence_length:]


def get_data(dataset, sensors, sequence_length, alpha, threshold):
    dir_path = './data/'
    train = pd.read_csv(dir_path + f'train_{dataset}.txt', sep=r'\s+', header=None)
    test = pd.read_csv(dir_path + f'test_{dataset}.txt', sep=r'\s+', header=None)
    y_test = pd.read_csv(dir_path + f'RUL_{dataset}.txt', sep=r'\s+', header=None,
                         names=['RemainingUsefulLife'])

    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i+1}' for i in range(21)]
    col_names = index_names + setting_names + sensor_names

    train.columns = col_names
    test.columns = col_names

    train = add_remaining_useful_life(train)
    train['RUL'] = train['RUL'].clip(upper=threshold)

    drop_sensors = [s for s in sensor_names if s not in sensors]
    X_train = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test = add_operating_condition(test.drop(drop_sensors, axis=1))

    X_train, X_test = condition_scaler(X_train, X_test, sensors)
    X_train = exponential_smoothing(X_train, sensors, 0, alpha)
    X_test = exponential_smoothing(X_test, sensors, 0, alpha)

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    for tr, va in gss.split(X_train['unit_nr'].unique(),
                            groups=X_train['unit_nr'].unique()):
        tr_units = X_train['unit_nr'].unique()[tr]
        va_units = X_train['unit_nr'].unique()[va]

    x_train = gen_data_wrapper(X_train, sequence_length, sensors, tr_units)
    y_train = gen_label_wrapper(X_train, sequence_length, ['RUL'], tr_units)
    x_val = gen_data_wrapper(X_train, sequence_length, sensors, va_units)
    y_val = gen_label_wrapper(X_train, sequence_length, ['RUL'], va_units)

    x_test = np.concatenate([
        list(gen_test_data(X_test[X_test['unit_nr'] == u],
                           sequence_length, sensors, -99.))
        for u in X_test['unit_nr'].unique()
    ]).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test['RemainingUsefulLife']


# --------------------------------------- TRAINING CALLBACKS ----------------------------------------
class save_latent_space_viz(Callback):
    def __init__(self, model, data, target):
        self.model = model
        self.data = data
        self.target = target
        self.best_val_loss = 1e9

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
            encoder = self.model.layers[0]
            viz_latent_space(encoder, self.data, self.target, epoch, True, False)

def get_callbacks(model, data, target):
    return [
        EarlyStopping(monitor='val_loss', patience=30, verbose=1),
        ModelCheckpoint('./checkpoints/checkpoint',
                        save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir='./logs'),
        save_latent_space_viz(model, data, target)
    ]

def viz_latent_space(encoder, data, targets=[], epoch='Final',
                     save=False, show=True):
    z, _, _ = encoder.predict(data)
    plt.figure(figsize=(8, 10))
    plt.scatter(z[:, 0], z[:, 1], c=targets if len(targets) else None)
    plt.colorbar()
    if save:
        plt.savefig(f'./images/latent_space_epoch{epoch}.png')
    if show:
        plt.show()
    return z


# ----------------------------------------- FIND OPTIMAL LR -----------------------------------------
class LRFinder:
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        lr = get_lr(self.model.optimizer)
        self.lrs.append(lr)

        loss = logs['loss']
        self.losses.append(loss)

        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        self.best_loss = min(self.best_loss, loss)
        set_lr(self.model.optimizer, lr * self.lr_mult)

    def find(self, x_train, y_train, start_lr, end_lr,
             batch_size=64, epochs=1, **kw_fit):

        N = x_train.shape[0]
        self.lr_mult = (end_lr / start_lr) ** (1.0 / (epochs * N / batch_size))

        initial_weights = self.model.get_weights()
        original_lr = get_lr(self.model.optimizer)

        set_lr(self.model.optimizer, start_lr)
        callback = LambdaCallback(
            on_batch_end=lambda b, l: self.on_batch_end(b, l)
        )

        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[callback],
                       **kw_fit)

        self.model.set_weights(initial_weights)
        set_lr(self.model.optimizer, original_lr)


# --------------------------------------------- RESULTS ---------------------------------------------
def get_model(path):
    model = load_model(path, compile=False)
    return model.layers[1], model.layers[2]

def evaluate(y_true, y_hat, label='test'):
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    r2 = r2_score(y_true, y_hat)
    print(f'{label} RMSE={rmse:.4f}, R2={r2:.4f}')

def score(y_true, y_hat):
    res = 0
    for t, p in zip(y_true, y_hat):
        diff = p - t
        res += np.exp(-diff / 10) - 1 if diff < 0 else np.exp(diff / 13) - 1
    print("PHM score:", res)

def results(path, x_train, y_train, x_test, y_test):
    encoder, regressor = get_model(path)
    z_train = viz_latent_space(encoder, x_train, y_train)
    z_test = viz_latent_space(encoder, x_test, y_test)
    evaluate(y_train, regressor.predict(z_train), 'train')
    evaluate(y_test, regressor.predict(z_test), 'test')
    score(y_test, regressor.predict(z_test))
