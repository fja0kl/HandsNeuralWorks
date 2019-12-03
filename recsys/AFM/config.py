'''
杂项:路径设置etc; 处理特征列表
'''

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"

SUB_DIR = "output"


NUM_SPLITS = 3
RANDOM_SEED = 2017

"""
feature ps_car_01_cat, value list: [10 11  7  6  9  5  4  8  3  0  2  1 -1]
feature ps_car_02_cat, value list: [1 0]
feature ps_car_03_cat, value list: [-1  0  1]
feature ps_car_04_cat, value list: [0 1 8 9 2 6 3 7 4 5]
feature ps_car_05_cat, value list: [ 1 -1  0]
feature ps_car_06_cat, value list: [ 4 11 14 13  6 15  3  0  1 10 12  9 17  7  8  5  2 16]
feature ps_car_07_cat, value list: [ 1 -1  0]
feature ps_car_08_cat, value list: [0 1]
feature ps_car_09_cat, value list: [ 0  2  3  1 -1  4]
feature ps_car_10_cat, value list: [1 0 2]
feature ps_car_11, value list: [2 3 1 0]
feature ps_car_11_cat, value list: [ 12  19  60 104  82  99  30  68  20  36 101 103  41  59  43  64  29  95
  24   5  28  87  66  10  26  54  32  38  83  89  49  93   1  22  85  78
  31  34   7   8   3  46  27  25  61  16  69  40  76  39  88  42  75  91
  23   2  71  90  80  44  92  72  96  86  62  33  67  73  77  18  21  74
  37  48  70  13  15 102  53  65 100  51  79  52  63  94   6  57  35  98
  56  97  55  84  50   4  58   9  17  11  45  14  81  47]
feature ps_ind_01, value list: [2 1 5 0 4 3 6 7]
feature ps_ind_02_cat, value list: [ 2  1  4  3 -1]
feature ps_ind_03, value list: [ 5  7  9  2  0  4  3  1 11  6  8 10]
feature ps_ind_04_cat, value list: [ 1  0 -1]
feature ps_ind_05_cat, value list: [ 0  1  4  3  6  5 -1  2]
feature ps_ind_06_bin, value list: [0 1]
feature ps_ind_07_bin, value list: [1 0]
feature ps_ind_08_bin, value list: [0 1]
feature ps_ind_09_bin, value list: [0 1]
feature ps_ind_10_bin, value list: [0 1]
feature ps_ind_11_bin, value list: [0 1]
feature ps_ind_12_bin, value list: [0 1]
feature ps_ind_13_bin, value list: [0 1]
feature ps_ind_14, value list: [0 1 2 3]
feature ps_ind_15, value list: [11  3 12  8  9  6 13  4 10  5  7  2  0  1]
feature ps_ind_16_bin, value list: [0 1]
feature ps_ind_17_bin, value list: [1 0]
feature ps_ind_18_bin, value list: [0 1]
"""
# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    # 'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    # 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
    # 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    # 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
    # 'ps_car_10_cat', 'ps_car_11_cat',
]

NUMERIC_COLS = [
    # # binary
    # "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin",
    # "ps_ind_09_bin", "ps_ind_10_bin", "ps_ind_11_bin",
    # "ps_ind_12_bin", "ps_ind_13_bin", "ps_ind_16_bin",
    # "ps_ind_17_bin", "ps_ind_18_bin",
    # "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    # "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin",
    # numeric
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",

    # feature engineering
    "missing_feat", "ps_car_13_x_ps_reg_03",
]

IGNORE_COLS = [
    "id", "target",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]
