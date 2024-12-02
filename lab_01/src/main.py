import math
import numpy as np
from Interval_new import Interval
from decimal import Decimal, getcontext

getcontext().prec = 50

def det(mid_matrix, alpha):
  i1 = Interval(Decimal(mid_matrix[0][0]) - alpha, Decimal(mid_matrix[0][0]) + alpha)
  i2 = Interval(Decimal(mid_matrix[0][1]) - alpha, Decimal(mid_matrix[0][1]) + alpha)
  i3 = Interval(Decimal(mid_matrix[1][0]) - alpha, Decimal(mid_matrix[1][0]) + alpha)
  i4 = Interval(Decimal(mid_matrix[1][1]) - alpha, Decimal(mid_matrix[1][1]) + alpha)
  det = i1 * i4 - i2 * i3
  return det

def find_upper_bound(mid_matrix):
  alpha = Decimal(1)

  while 0 not in det(mid_matrix, alpha):
    alpha = Decimal(math.exp(float(alpha)))

  return alpha

def find_alpha(mid_matrix, eps=10e-18):
  a = Decimal(0)
  b = find_upper_bound(mid_matrix)
  k = 0

  alphas = []
  dets = []

  while b - a > eps:
    mid = (a + b) / 2

    alphas.append(mid)
    dets.append(det(mid_matrix, mid))

    if 0 in det(mid_matrix, mid):
      b = mid
    else:
      a = mid
    print("k = " + str(k) + "; alpha = " + str(mid) + "; det = " + str(det(mid_matrix, mid)))

    k += 1


  alpha = (a + b) / 2

  alphas.append(alpha)
  dets.append(det(mid_matrix, alpha))
  return alphas, dets

if __name__ == '__main__':
  mid_matrix = np.array([
    [Decimal('1.05'), Decimal('0.95')],
    [Decimal('1.0'), Decimal('1.0')],
  ])

  alphas, dets = find_alpha(mid_matrix, 0.0000001)

  print(len(alphas), alphas[-1])
