from time import time as t
from tensorflow import function as tf_function_decorator
from tensorflow import GradientTape
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


def train(
        model=None,
        train_ds=None,
        val_ds=None,
        epochs: int = 1,
        batch_size: int = 1,
        saved_path: str = ''
):
    if model is not Model:
        raise ValueError(f'model must be of type keras.Model but got {type(model)}.')
    if train_ds is None:
        raise ValueError('train_ds must be passed but got None.')

    if saved_path != '' and saved_path[-1] != '\\':
        saved_path += '\\'

    # Instantiate an optimizer.
    optimizer = SGD(learning_rate=1e-3)  # TODO: change optimizer

    # Instantiate a loss function.
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)  # TODO: change loss function

    # Prepare the training dataset.
    train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    if val_ds is not None:
        val_ds = val_ds.batch(batch_size)

    # Prepare the metrics.
    train_metric = SparseCategoricalAccuracy()  # TODO: change training metrics
    if val_ds is not None:
        val_metric = SparseCategoricalAccuracy()  # TODO: change validation metrics
    else:
        val_metric = None

    @tf_function_decorator
    def train_step(x, y):
        with GradientTape() as tape:
            preds = model(x, training=True)
            loss_value = loss_fn(y, preds)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_metric.update_state(y, preds)
        return loss_value

    @tf_function_decorator
    def test_step(x, y):
        val_preds = model(x, training=False)
        val_metric.update_state(y, val_preds)

    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")
        start_time = t()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            loss = train_step(x_batch_train, y_batch_train)

            # Log every 200 batches.
            if step % 200 == 0:
                print(f"Training loss (for one batch) at step {step}: {round(float(loss), 4)}")
                print(f"Seen so far: {(step + 1) * batch_size} samples")

        # Display metrics at the end of each epoch.
        train_acc = train_metric.result()
        print(f"Training acc over epoch: {round(float(train_acc), 4)}")

        # Reset training metrics at the end of each epoch
        train_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        if val_ds is not None:
            for x_batch_val, y_batch_val in val_ds:
                test_step(x_batch_val, y_batch_val)

            val_acc = val_metric.result()
            val_metric.reset_states()
            print(f"Validation acc: {round(float(val_acc), 4)}")

        print(f"Time taken: {round(t() - start_time, 2)}s")

        model.save(saved_path+'model.h5')
