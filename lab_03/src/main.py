#%%
import struct
import numpy as np
import intvalpy as ip
from functools import cmp_to_key
#%%
def read_bin_file_with_numpy(file_path):
  with open(file_path, 'rb') as f:
    header_data = f.read(256)
    side, mode, frame_count = struct.unpack('<BBH', header_data[:4])

    frames = []
    point_dtype = np.dtype('<8H')

    for _ in range(frame_count):
      frame_header_data = f.read(16)
      stop_point, timestamp = struct.unpack('<HL', frame_header_data[:6])
      frame_data = np.frombuffer(f.read(1024 * 16), dtype=point_dtype)
      frames.append(frame_data)

    return np.array(frames)
#%%
def convert_to_voltage(data):
  return data / 16384.0 - 0.5
#%%
def are_intersected(x, y):
  sup = y.a if x.a < y.a else x.a
  inf = x.b if x.b < y.b else y.b
  return sup - inf <= 1e-15
#%%
def are_adjusted_to_each_other(x, y):
  return x.b == y.a or y.b == x.a
#%%
def merge_intervals(x, y):
  return ip.Interval(min(x.a, y.a), max(x.b, y.b))
#%%
def mode(x):
  if len(x) == 0:
    return []

  edges = sorted({x_i.a for x_i in x}.union({x_i.b for x_i in x}))
  z = [ip.Interval(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
  mu = [sum(1 for x_i in x if z_i in x_i) for z_i in z]

  max_mu = max(mu)
  K = [index for index, element in enumerate(mu) if element == max_mu]

  m = [z[k] for k in K]
  merged_m = []

  current_interval = m[0]

  for next_interval in m[1:]:
    if are_intersected(current_interval, next_interval) or are_adjusted_to_each_other(current_interval, next_interval):
      current_interval = merge_intervals(current_interval, next_interval)
    else:
      merged_m.append(current_interval)
      current_interval = next_interval

  merged_m.append(current_interval)

  return merged_m
#%%
def med_k(x):
  starts = [float(interval.a) for interval in x]
  ends = [float(interval.b) for interval in x]
  return ip.Interval(np.median(starts), np.median(ends))
#%%
def med_p(x):
  x = sorted(x, key=cmp_to_key(lambda x, y: (x.a + x.b) / 2 - (y.a + y.b) / 2))

  mid_index = len(x) // 2

  if len(x) % 2 == 0:
    return (x[mid_index - 1] + x[mid_index]) / 2

  return x[mid_index]
#%%
def jaccard_index(*args):
  if len(args) == 1:
    x = args[0]
    left_edges = [interval.a for interval in x]
    right_edges = [interval.b for interval in x]

    return (min(right_edges) - max(left_edges)) / (max(right_edges) - min(left_edges))
  elif len(args) == 2:
    x = args[0]
    y = args[1]

    if isinstance(x, ip.ClassicalArithmetic) and isinstance(y, ip.ClassicalArithmetic):
      return (min(x.b, y.b) - max(x.a, y.a)) / (max(x.b, y.b) - min(x.a, y.a))
    else:
      results = []

      for x_i, y_i in zip(x, y):
        result = (min(x_i.b, y_i.b) - max(x_i.a, y_i.a)) / (max(x_i.b, y_i.b) - min(x_i.a, y_i.a))
        results.append(result)

      return np.array(results)
  else:
    raise ValueError("Wrong number of arguments")
#%%
def ternary_search(f, left, right, eps):
  while right - left > eps:
    m1 = left + (right - left) / 3
    m2 = right - (right - left) / 3

    if f(m1) < f(m2):
      left = m1
    else:
      right = m2

  return (left + right) / 2
#%%
def estimate_a(a):
  return np.mean(jaccard_index(x_voltage_int_flatten + a, y_voltage_int_flatten))

def estimate_t(t):
  return np.mean(jaccard_index(x_voltage_int_flatten * t, y_voltage_int_flatten))

def estimate_a_mode(a):
  return np.mean(jaccard_index(mode(x_voltage_int_flatten + a), mode(y_voltage_int_flatten)))

def estimate_t_mode(t):
  x = mode(x_voltage_int_flatten * t)
  x_idx = len(x) // 2
  x = x[x_idx]

  y = mode(y_voltage_int_flatten)
  y_idx = len(y) // 2
  y = y[y_idx]

  return np.mean(jaccard_index(x, y))

def estimate_a_med_p(a):
  return np.mean(jaccard_index(med_p(x_voltage_int_flatten + a), med_p(y_voltage_int_flatten)))

def estimate_t_med_p(t):
  return np.mean(jaccard_index(med_p(x_voltage_int_flatten * t), med_p(y_voltage_int_flatten)))

def estimate_a_med_k(a):
  return np.mean(jaccard_index(med_k(x_voltage_int_flatten + a), med_k(y_voltage_int_flatten)))

def estimate_t_med_k(t):
  return np.mean(jaccard_index(med_k(x_voltage_int_flatten * t), med_k(y_voltage_int_flatten)))
#%%
def scalar_to_interval(x, rad):
  return ip.Interval(x - rad, x + rad)
scalar_to_interval_vec = np.vectorize(scalar_to_interval)
#%%
x_data = read_bin_file_with_numpy('-0.205_lvl_side_a_fast_data.bin')
y_data = read_bin_file_with_numpy('0.225_lvl_side_a_fast_data.bin')
#%%
x_voltage = convert_to_voltage(x_data)
y_voltage = convert_to_voltage(y_data)
#%%
N = -14
rad = 2 ** N

x_voltage_int = scalar_to_interval_vec(x_voltage, rad)
y_voltage_int = scalar_to_interval_vec(y_voltage, rad)
#%%
x_voltage_int_flatten = x_voltage_int.flatten()
y_voltage_int_flatten = y_voltage_int.flatten()
#%%
# a_1 = ternary_search(estimate_a, 0, 1, 1e-3)
# print(a_1, estimate_a(a_1))
#
# #%%
# t_1 = ternary_search(estimate_t, -4, 0, 1e-3)
# print(t_1, estimate_t(t_1))
# #%%
# a_2 = ternary_search(estimate_a_mode, -4, 4, 1e-3)
# print(a_2, estimate_a_mode(a_2))
# #%%
# t_2 = ternary_search(estimate_t_mode, -4, 0, 1e-3)
# print(t_2, estimate_t_mode(t_2))
# #%%
# a_3 = ternary_search(estimate_a_med_p, -4, 4, 1e-3)
# print(a_3, estimate_a_med_p(a_3))
# #%%
# t_3 = ternary_search(estimate_t_med_p, -4, 0, 1e-3)
# print(t_3, estimate_t_med_p(t_3))
# #%%
# a_4 = ternary_search(estimate_a_med_k, -4, 4, 1e-3)
# print(a_4, estimate_a_med_k(a_4))
# #%%
# t_4 = ternary_search(estimate_t_med_k, -4, 0, 1e-3)
# print(t_4, estimate_t_med_k(t_4))
#%%
def confidence_interval(param_estimates, alpha=0.05):
    """
    Рассчитывает точечную оценку и доверительный интервал для параметров.

    Параметры:
        param_estimates: массив оценок параметра (например, a или t)
        alpha: уровень значимости (по умолчанию 0.05)

    Возвращает:
        точечная_оценка, (нижняя_граница, верхняя_граница)
    """
    mean_estimate = np.mean(param_estimates)
    std_error = np.std(param_estimates, ddof=1) / np.sqrt(len(param_estimates))
    z = np.norm.ppf(1 - alpha / 2)  # Квантиль стандартного нормального распределения
    lower_bound = mean_estimate - z * std_error
    upper_bound = mean_estimate + z * std_error
    return mean_estimate, (lower_bound, upper_bound)
#%%
results_a = []
results_a_mode = []
results_a_med_p = []
results_a_med_k = []
results_t = []
results_t_mode = []
results_t_med_p = []
results_t_med_k = []

# Генерируем подвыборки и проводим тернарный поиск
# for _ in range(50):  # 100 подвыборок
indices = np.random.choice(len(x_voltage_int_flatten), size=len(x_voltage_int_flatten), replace=True)
x_sample = x_voltage_int_flatten[indices]
y_sample = y_voltage_int_flatten[indices]

# Оптимизация на подвыборках
opt_a = ternary_search(lambda a: -estimate_a(a), left=0, right=1, eps=1e-1)  # Пример границ
opt_t = ternary_search(lambda t: -estimate_t(t), left=-4, right=0, eps=1e-1)
opt_a_mode = ternary_search(lambda a: -estimate_a_mode(a), left=-4, right=4, eps=1e-1)
opt_t_mode = ternary_search(lambda t: -estimate_t_mode(t), left=-4, right=0, eps=1e-1)
opt_a_med_p = ternary_search(lambda a: -estimate_a_med_p(a), left=-4, right=4, eps=1e-1)
opt_t_med_p = ternary_search(lambda t: -estimate_t_med_p(t), left=-4, right=0, eps=1e-1)
opt_a_med_k = ternary_search(lambda a: -estimate_a_med_k(a), left=-4, right=4, eps=1e-1)
opt_t_med_k = ternary_search(lambda t: -estimate_t_med_k(t), left=-4, right=0, eps=1e-1)
results_a.append(opt_a)
results_t.append(opt_t)
results_a_mode.append(opt_a_mode)
results_t_mode.append(opt_t_mode)
results_a_med_p.append(opt_a_med_p)
results_t_med_p.append(opt_t_med_p)
results_a_med_k.append(opt_a_med_k)
results_t_med_k.append(opt_t_med_k)


# Рассчитываем доверительные интервалы
a_mean, a_conf_int = confidence_interval(results_a, alpha=0.05)
t_mean, t_conf_int = confidence_interval(results_t, alpha=0.05)

a_mean_mode, a_conf_int_mode = confidence_interval(results_a_mode, alpha=0.05)
t_mean_mode, t_conf_int_mode = confidence_interval(results_t_mode, alpha=0.05)

a_mean_med_p, a_conf_int_med_p = confidence_interval(results_a_med_p, alpha=0.05)
t_mean_med_p, t_conf_int_med_p = confidence_interval(results_t_med_p, alpha=0.05)

a_mean_med_k, a_conf_int_med_k = confidence_interval(results_a_med_k, alpha=0.05)
t_mean_med_k, t_conf_int_med_k = confidence_interval(results_t_med_k, alpha=0.05)

print(f"Точечная оценка a: {a_mean}, Доверительный интервал: {a_conf_int}")
print(f"Точечная оценка t: {t_mean}, Доверительный интервал: {t_conf_int}")

print(f"Точечная оценка a: {a_mean_mode}, Доверительный интервал: {a_conf_int_mode}")
print(f"Точечная оценка t: {t_mean_mode}, Доверительный интервал: {t_conf_int_mode}")

print(f"Точечная оценка a: {a_mean_med_p}, Доверительный интервал: {a_conf_int_med_p}")
print(f"Точечная оценка t: {t_mean_med_p}, Доверительный интервал: {t_conf_int_med_p}")

print(f"Точечная оценка a: {a_mean_med_k}, Доверительный интервал: {a_conf_int_med_k}")
print(f"Точечная оценка t: {t_mean_med_k}, Доверительный интервал: {t_conf_int_med_k}")