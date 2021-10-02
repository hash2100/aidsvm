#
# Copyright 2020 Honorius Galmeanu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions
# of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import unittest
import numpy as np
import timing as t

from kernel import PolyKernel, RbfKernel
from migration import *
from svm import SVM, SingularMatrixException, NoOppositeClassLeftException


WINDOW_SIZE = 1500


def unit_test_migration():
    Migration.test_unit()
    print('\n')


def unit_test_svm():
    SVM.test_masked_min()
    print('\n')


def unit_test_polynomial():
    print('=== Polynomial test_unit() begin ===')

    # store Xes by rows
    device = torch.device('cuda')
    # device = torch.device('cpu')
    vectors = Migration(x=torch.tensor([[0.0, 6.0], [4.0, -6.0]], dtype=torch.double, device=device),
                        y=torch.tensor([-1.0, 1.0], dtype=torch.double, device=device).reshape((-1, 1)),
                        c_init=0.5, window_size=10)

    svm = SVM(kernel_class=PolyKernel, window_size=10)
    svm.initialize(vectors)
    print(f'equation of the plane: {svm.compute_w().cpu().numpy()} * x + {svm._b.cpu().numpy()}')
    print(f'distance from [4, 4] to plane: {svm.g(torch.tensor([4.0, 4.0], dtype=torch.double, device=device), legit=True).cpu().numpy()}')

    svm.append(2, x=torch.tensor([5.0, 2.0], dtype=torch.double), y=torch.tensor([-1.0], dtype=torch.double), c_init=0.5)
    updated = svm.k
    recomputed = svm._kernel()
    assert torch.all(torch.tensor(updated == recomputed))

    svm.show_state(False)

    assert svm.h(2) == svm.h(torch.tensor([5.0, 2.0], dtype=torch.double), -1, legit=True)

    # see the h() value for the third point
    # if it is in Others set, for lambda = 0, its h() should be > 0
    print(f'new point h(): {svm.h(2)}')

    # compute h for all vectors
    svm.print_h()
    svm.print_sets()
    g = svm.g(vectors.x, legit=True)
    h = svm.compute_h()
    gc = (h + 1) * svm.y.squeeze(1)
    print(f'g(0) = {g[0]}')
    print(f'g(1) = {g[1]}')
    print(f'g(2) = {g[2]}')
    assert g[0] == gc[0]
    assert g[1] == gc[1]
    assert g[2] == gc[2]

    # iterate as long as last vector does not migrate in support set or lambda_c != 0 or C
    svm.learn(i=2, C=0.5)
    svm.print_h()
    svm.print_sets()

    print(f'\n----------------\n')

    # now unlearn vector 2; put vector 2 on last position in rest vectors
    svm.move_to_end(2)
    svm.update_kernel()

    svm.print_sets()
    svm.print_h()

    # iterate as long as lambda_c > 0 (last vector)
    svm.unlearn(i=2, C=0.)
    svm.print_h()
    svm.print_sets()

    print('=== Polynomial test_unit() end ===')


def reinit(x_train, y_train, device, C, window_size):
    vectors = Migration(x=x_train.double().to(device), y=y_train.double().to(device).reshape(-1, 1), c_init=C, window_size=window_size)

    #svm = SVM(kernel_class=PolyKernel)
    svm = SVM(kernel_class=RbfKernel, window_size=WINDOW_SIZE)
    svm.initialize(vectors, gamma=5.99134645435374)  # sine1

    print(f'distance from x[0] to plane: {svm.g(x_train[0].double().to(device), legit=True).cpu().numpy()}')
    print(f'distance from x[1] to plane: {svm.g(x_train[1].double().to(device), legit=True).cpu().numpy()}')
    print(svm.g(x_train.double().to(device), legit=True).cpu().numpy())
    print(f'y_train: {y_train}')

    # compute h for all vectors
    svm.print_h()
    svm.print_sets()

    svm.init_statistics()

    return svm, vectors


def append_last_vector(svm, v, c_param) -> bool:
    """
    Learns the last vector added to the rest set
    Does not learn it if it is too similar with existing ones or if it determines singular matrix exception

    Args:
        svm: the SVM object
        v: the vector
        c_param: regularization parameter

    Returns:
        True if the vector was learned or False if it was skipped
    """

    # first, check if contribution of this vector is identical
    # with other's vector already added
    res = svm.detect_similar(svm.compute_q())
    if res:
        svm.remove()
        print(f'>> removing {v} vector as duplicate')
        svm.update_kernel()
        svm.show_state(False)
        return False

    # learn most recent vector
    last = svm.rest_set[-1]
    assert last == v
    i, c = last, c_param

    # print(f'>> learn {i} vector to {c}')
    try:
        svm.learn(i, c)
    except SingularMatrixException as e:
        print(f'>> exception: {e}')
        print(f'>> removing {last} vector as singular')
        c = 0.0
        svm.unlearn(i, c)
        return False

    return True


def remove_first_vector(svm, c_param) -> bool:
    """
    Removes the first vector from the shifting window

    Args:
        svm: the SVM object
        c_param: regularization parameter

    Returns:
        True if unlearned, False if it was not unlearned due to nonexistence or being a single representative
    """
    # unlearn first sample
    first = min(svm.support_set) if svm.rest_set is None or len(svm.rest_set) == 0 \
        else min(min(svm.rest_set), min(svm.support_set))

    # print(f'>> unlearn {i} vector to {c}')
    i, c = first, 0.0

    if svm.prevent_unlearn(i):
        print(f'>> cannot unlearn {i}, it is the only of its class')
        return False

    if i not in svm.support_set and i not in svm.rest_set:
        print(f'>> vector {i} already unlearned')
        return False

    try:
        svm.unlearn(i, c)
    except NoOppositeClassLeftException as e:
        c = c_param   # as previous
        svm.learn(i, c, revert=True)

    return True


def get_positions(svm, window):
    return [svm.get_pos(i) for i in window]


def compute_max_mean_discrepancy(svm, window, split_idx):
    assert 0 < split_idx < len(window)

    split0 = get_positions(svm, window[:split_idx])
    split1 = get_positions(svm, window[split_idx:])

    exp0 = svm.k[split0][:, split0]
    exp1 = svm.k[split1][:, split1]
    exp2 = svm.k[split0][:, split1]
    max_mean_discrepancy = exp0.mean() + exp1.mean() - 2 * exp2.mean()

    print(max_mean_discrepancy)
    return max_mean_discrepancy


def evaluate_window(matches, delta=0.1):
    mu0, mu1, sizes = compute_partitions(matches, skip_margin=21)

    n = len(matches)
    m = 1 / (1. / sizes.float() + 1. / (n - sizes).float())
    delta_p = delta / n
    term = 1. / m * np.log(2. / delta_p)
    var = 0.25
    term1 = torch.sqrt(var * term)
    term2 = term / 3
    eps_cut = term1 + term2
    diff = torch.abs(mu0 - mu1)
    cuts = torch.nonzero(diff >= eps_cut)

    return cuts


def compute_partitions(matches, skip_margin):
    n = len(matches)
    partitions = matches.repeat(n - 2 * skip_margin + 1, 1)
    w0 = partitions.tril(skip_margin - 1)
    w1 = partitions.triu(skip_margin)

    sizes = torch.arange(skip_margin, n - skip_margin + 1, device=matches.device)
    assert len(sizes) == n - 2 * skip_margin + 1
    w0_failures = (sizes - w0.sum(dim=1)).float()
    w1_failures = (n - sizes - w1.sum(dim=1)).float()
    mu0, mu1 = w0_failures / sizes, w1_failures / (n - sizes)

    return mu0, mu1, sizes


class WindowTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_compute_partitions(self):
        matches = torch.arange(10)
        w0_f, w1_f, _ = compute_partitions(matches, 3)
        n, sizes = 10, torch.tensor([3., 4, 5, 6, 7])

        assert torch.all(w0_f == (sizes - torch.tensor([3., 6, 10, 15, 21])) / sizes)
        assert torch.all(w1_f == (n - sizes - torch.tensor([42., 39, 35, 30, 24])) / (n - sizes))

    def test_compute_partitions_real(self):
        matches = torch.arange(10)
        matches = (matches % 3 < 2)
        w0_f, w1_f, _ = compute_partitions(matches, 3)
        n, sizes = 10, torch.tensor([3., 4, 5, 6, 7])

        assert torch.all(w0_f == torch.tensor([1., 1, 1, 2, 2]) / sizes)
        assert torch.all(w1_f == torch.tensor([2., 2, 2, 1, 1]) / (n - sizes))


DRIFT_DETECTION = True


def main():
    C = 10.

    # device = torch.device('cpu')
    device = torch.device('cuda')

    d, m = torch.load('../data/sine1_101.pt'), 2
    x_train, y_train = d['x'][:m], d['y'][:m]

    svm, vectors = reinit(x_train, y_train, device, C, window_size=WINDOW_SIZE)

    # maximum number of vectors to be kept in the window
    window_width = WINDOW_SIZE

    svm.debug = False
    svm.init_statistics()

    # test in advance on set_len samples
    set_len = 100

    file = open('running.csv', 'w')
    set_file = open('current_set.txt', 'w')
    winlen_file = None
    if DRIFT_DETECTION:
        winlen_file = open('winlen.csv', 'w')

    # first, get a version that dynamically adjusts the shifting window
    for v in range(2, d['x'].shape[0]):
        x_set, y_set = d['x'][v:v+set_len].double().to(device), d['y'][v:v+set_len].double().to(device)
        x, y = d['x'][v].double().to(device), d['y'][v].double().to(device).unsqueeze(0)

        # if v > 1000:
        #     break

        if v == window_width:
            print('Reached window width')

        if v % 100 == 0:
            print(f'>> adding vector {v} to window')

        # before learning the vector, show how it is classified
        #svm.collect_statistics(file, v, x_set, y_set)

        # append the last vector
        svm.append(v, x, y, c_init=C)
        svm.show_state(False)

        # before learning the vector, show how it is classified
        svm.collect_statistics(file, v, x_set, y_set)

        # print current set
        all_set = svm.support_set + svm.rest_set
        print(f'{v}, {sorted(all_set)}', file=set_file)

        # effectively learns the vector
        append_last_vector(svm, v, C)
        svm.show_state(False)

        # test whether there are enough samples
        all_set = svm.support_set + svm.rest_set

        # set a minimum limit for the window
        window = sorted(all_set)
        if len(window) < 50:
            continue

        if DRIFT_DETECTION:
            window_t = torch.tensor(window)
            matches = svm.compute_matches(d['x'][window_t].double().to(device), d['y'][window_t].double().to(device))
            cuts = evaluate_window(matches, delta=0.1)

            while len(cuts) > 0:
                print(f'== cuts is non-empty, removing first vector')
                remove_first_vector(svm, C)

                all_set = svm.support_set + svm.rest_set
                window_t = torch.tensor(sorted(all_set))
                matches = svm.compute_matches(d['x'][window_t].double().to(device), d['y'][window_t].double().to(device))
                cuts = evaluate_window(matches, delta=0.1)
                print(f'== current window len: {len(matches)}')

            n = len(matches)
            print(f'{v}, {n}', file=winlen_file)
            print(f'window len: {len(window)}')

        # we are not removing the first vector unless the window is full
        if len(all_set) < window_width:
            continue

        remove_first_vector(svm, C)

        # print(f'window len: {len(all_set)}, interval [{min(all_set)}, {max(all_set)}]')

    file.close()
    set_file.close()
    if DRIFT_DETECTION:
        winlen_file.close()

    svm.show_state(force=True)


def main_tests():
    unit_test_migration()
    unit_test_svm()
    unit_test_polynomial()


if __name__ == '__main__':
    main()
    # unittest.main()
    # main_tests()
