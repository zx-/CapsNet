import tensorflow as tf
import loss
import data
from capsnet import build_graph
import tensorflow.contrib.slim as slim
import argparse


def capsnet_model_fn(features, labels, mode, params):
    x = features
    capsules, reconstruction = build_graph(x, labels)

    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.norm(capsules, axis=-1)
        predictions = {
            'capsules': capsules,
            'reconstruction': reconstruction,
            'probabilities': probabilities,
            'classes': tf.argmax(probabilities, axis=-1),
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    total_loss, margin_loss, rec_loss = loss.total_loss(capsules, labels, reconstruction, x)
    accuracy = loss.accuracy(capsules, labels)

    with tf.name_scope('metrics'):
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('margin_loss', margin_loss)
        tf.summary.scalar('rec_loss', rec_loss)

    eval_metrics = {
        'metrics/accuracy': tf.metrics.mean(accuracy, name='acc_mean'),
        'metrics/margin_loss': tf.metrics.mean(margin_loss, name='ml_mean'),
        'metrics/rec_loss': tf.metrics.mean(rec_loss, name='rl_mean'),
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=eval_metrics)

    opt = tf.train.AdamOptimizer(0.03)
    # training_operation = slim.learning.create_train_op(total_loss, opt, summarize_gradients=False,
    #                                                    global_step=tf.train.get_global_step())

    t_op = opt.minimize(total_loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=t_op)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Capsnet over MNIST dataset using estimator.
        Stores tensorboard data in tmp/log/<name>
        """)
    parser.add_argument('-e', '--epochs', action="store", dest="epochs", type=int, default=5,
                        help="Number of epochs to run.")
    parser.add_argument('-b', '--batch', action="store", dest="batch", type=int, default=64, help="Batch size.")
    parser.add_argument('-n', '--name', action="store", dest="name", type=str, default='train_it_1',
                        help="Run name to use in tensorboard.")
    arguments = parser.parse_args()

    capsnet_classifier = tf.estimator.Estimator(
        model_fn=capsnet_model_fn,
        model_dir=f'tmp/log/{arguments.name}',
        params={},
        config=tf.estimator.RunConfig(
            save_summary_steps=100,
            keep_checkpoint_max=5
        )
    )

    train, test = data.create_datasets()

    for i in range(arguments.epochs):
        capsnet_classifier.train(
            input_fn=data.data_input_fn(train, batch_size=arguments.batch, shuffle=True))

        capsnet_classifier.evaluate(
            input_fn=data.data_input_fn(test, batch_size=arguments.batch))


