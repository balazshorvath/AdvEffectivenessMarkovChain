"""
    A simple simulation for the
    "two cups, different ratios, drawing black, stay on first cup, white draw from the other one".

    IN: iterations b1, w1, b2, w2
"""
from random import randint
from sys import argv

import pylab as pl

black1 = int(argv[2])
total1 = black1 + int(argv[3])
black2 = int(argv[4])
total2 = black2 + int(argv[5])

overall_ratio = float(total1 - black1 + total2 - black2) / float(black1 + black2)

current = 1
black_picks = 1
white_picks = 0
interval = range(1, int(argv[1]) + 1)
black_white_ratios = []  # np.zeros(len(interval))

pl.ion()
pl.figure()
# bar_chart = fig.add_subplot(2, 1, 1)
# convergence_chart = fig.add_subplot(2, 1, 1)

for i in interval:
    if current == 1:
        pick = randint(0, total1)
        if pick > black1:
            current = 2
            white_picks += 1
        else:
            black_picks += 1
    else:
        pick = randint(0, total2)
        if pick <= black2:
            black_picks += 1
            current = 1
        else:
            white_picks += 1

    black_white_ratios.append(white_picks / black_picks)
    if i % 100 == 0:
        # The constant re-plotting without clearing the data is bad,
        # but if I clear it just a couple of times, the progress is smoother
        pl.clf()
    if i % 10 == 0:
        pl.subplot(2, 1, 1)
        pl.grid(True)
        pl.bar(range(2), [black_picks, white_picks], align='center', color=['Black', 'Gray'],
               tick_label=['Black', 'White'])
        pl.subplot(2, 1, 2)
        pl.grid(True)
        pl.plot(interval[:i], black_white_ratios, color='Blue', linestyle='-')
        pl.axhline(y=overall_ratio, color='Green', linestyle='-')
        pl.pause(0.0001)
    # pl.show()

pl.waitforbuttonpress()
