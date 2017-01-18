"""
Estimator for GAN
"""

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import LabeledImageDataset
from chainer.datasets import get_cross_validation_datasets_random

from util.file_accessor import get_teacher_data

class Estimator(chainer.Chain):
    def __init__(self):
        super(Estimator, self).__init__(
            l1=L.Linear(3147264, 10000),
            l2=L.Linear(10000, 100),
            l3=L.Linear(100, 1)
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

model = L.Classifier(Estimator())
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# train, test = chainer.datasets.get_mnist()

if __name__ == '__main__':
    # 引数指定
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str)
    parser.add_argument('--label', '-l', type=str)
    args = parser.parse_args()

    teacher_data = get_teacher_data(args.path, args.label)
    teacher = LabeledImageDataset(teacher_data)
    train, test = get_cross_validation_datasets_random(teacher, 1)

    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (100, 'epoch'), out="result")
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    print('Start training')
    trainer.run()
