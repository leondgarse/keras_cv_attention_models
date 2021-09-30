import numpy as np
from tqdm import tqdm
from keras_cv_attention_models.imagenet import data

def eval(model, data_name="imagenet2012", input_shape=None, batch_size=64, central_fraction=1.0, mode='tf'):
    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape
    _, test_dataset, _, _, _ = data.init_dataset(data_name, input_shape=input_shape, batch_size=batch_size, central_fraction=central_fraction, mode=mode)

    y_true, y_pred_top_1, y_pred_top_5 = [], [], []
    for img_batch, true_labels in tqdm(test_dataset, "Evaluating", total=len(test_dataset)):
        predicts = model(img_batch).numpy()
        pred_args = predicts.argsort(-1)
        y_pred_top_1.extend(pred_args[:, -1])
        y_pred_top_5.extend(pred_args[:, -5:])
        y_true.extend(np.array(true_labels).argmax(-1))
    y_true, y_pred_top_1, y_pred_top_5 = np.array(y_true), np.array(y_pred_top_1), np.array(y_pred_top_5)
    accuracy_1 = np.sum(y_true == y_pred_top_1) / y_true.shape[0]
    accuracy_5 = np.sum([ii in jj for ii, jj in zip(y_true, y_pred_top_5)]) / y_true.shape[0]
    print(">>>> Accuracy top1:", accuracy_1, "top5:", accuracy_5)
    return y_true, y_pred_top_1, y_pred_top_5
